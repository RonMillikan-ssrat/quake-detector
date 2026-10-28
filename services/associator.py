"""
associator -- reads all picks, runs PyOcto over the last ASSOC_S seconds, reports each
event once it is SETTLE_S old, writes picks and events to Postgres, and publishes
events to the 'events' stream.

Singleton (it must see every pick). Unlike phase3_live.py, association runs only when
new picks have arrived or an unsettled event is waiting -- in quiet periods it idles.

Metrics:
  qd_assoc_seconds             PyOcto time per run (histogram)
  qd_events_total              events reported
  qd_detection_latency_seconds report time - origin time (histogram)
  qd_picks_held                picks in the association window
"""
import time

import pandas as pd
import psycopg
import pyocto
from obspy import UTCDateTime
from obspy.geodetics import gps2dist_azimuth
from prometheus_client import Counter, Gauge, Histogram, start_http_server

from services.common import Bus, beat, cfg, load_stations

ASSOC = Histogram("qd_assoc_seconds", "PyOcto seconds per run",
                  buckets=(0.1, 0.25, 0.5, 1, 2, 5, 10, 30))
EVENTS = Counter("qd_events_total", "Events reported")
LATENCY = Histogram("qd_detection_latency_seconds", "Origin to report",
                    buckets=(20, 30, 40, 50, 60, 90, 120, 180, 300))
HELD = Gauge("qd_picks_held", "Picks in association window")

SCHEMA = """
create table if not exists picks (station text, phase text, time_utc timestamptz,
  prob real, found_utc timestamptz);
create table if not exists events (origin_utc timestamptz, lat real, lon real,
  depth_km real, n_sta int, n_picks int, reported_utc timestamptz, latency_s real);
"""


def connect_db(dsn, timeout=120):
    t_end = time.time() + timeout
    while True:
        try:
            con = psycopg.connect(dsn, autocommit=True)
            con.execute(SCHEMA)
            return con
        except psycopg.OperationalError:
            if time.time() > t_end:
                raise
            print("waiting for postgres...", flush=True)
            time.sleep(3)


def main():
    c = cfg()
    start_http_server(c.metrics_port)
    bus = Bus(c.redis_url)
    bus.wait()
    db = connect_db(c.pg_dsn)

    sdf = pd.DataFrame(load_stations(bus))
    sdf["id"] = sdf["net"] + "." + sdf["sta"]
    sdf = sdf.rename(columns={"lat": "latitude", "lon": "longitude", "elev_m": "elevation"})
    assoc = pyocto.OctoAssociator.from_area(
        lat=(c.box["minlatitude"], c.box["maxlatitude"]),
        lon=(c.box["minlongitude"], c.box["maxlongitude"]),
        zlim=(0.0, 20.0),
        velocity_model=pyocto.VelocityModel0D(p_velocity=c.vp, s_velocity=c.vs,
                                              tolerance=c.tol),
        time_before=80.0, n_picks=c.n_picks, n_p_and_s_picks=c.n_ps)
    assoc.transform_stations(sdf)
    print(f"associator: {len(sdf)} stations", flush=True)

    picks, reported, last = [], [], "$"
    new_since_run, pending = False, False
    next_t = time.time() + c.step_s
    while True:
        last, entries = bus.read("picks", last, block_ms=(next_t - time.time()) * 1000)
        for f in entries:
            p = dict(station=f[b"station"].decode(), phase=f[b"phase"].decode(),
                     time=float(f[b"time"]), probability=float(f[b"prob"]))
            picks.append(p)
            db.execute("insert into picks values (%s,%s,to_timestamp(%s),%s,to_timestamp(%s))",
                       (p["station"], p["phase"], p["time"], p["probability"],
                        float(f[b"found"])))
            new_since_run = True
        if time.time() < next_t:
            continue
        next_t = max(next_t + c.step_s, time.time() + 1)
        beat()

        now = time.time()
        picks = [q for q in picks if q["time"] > now - 2 * c.assoc_s]
        recent = [q for q in picks if q["time"] > now - c.assoc_s]
        HELD.set(len(recent))
        if not (new_since_run or pending) or len(recent) < c.n_picks:
            continue
        new_since_run, pending = False, False

        tic = time.time()
        events, asg = assoc.associate(
            pd.DataFrame(recent)[["station", "time", "phase", "probability"]], sdf)
        ASSOC.observe(time.time() - tic)
        if not len(events):
            continue
        assoc.transform_events(events)
        for e in events.itertuples():
            age = now - e.time
            if age < c.settle_s:
                pending = True          # re-run next cycle even if no new picks arrive
                continue
            if age > c.assoc_s - 90:
                continue
            if any(abs(e.time - r[0]) < 3 and
                   gps2dist_azimuth(e.latitude, e.longitude, r[1], r[2])[0] < 15000
                   for r in reported):
                continue
            reported.append((e.time, e.latitude, e.longitude))
            reported = [r for r in reported if r[0] > now - 2 * c.assoc_s]
            ea = asg[asg["event_idx"] == e.idx]
            n_sta, n_p = int(ea["station"].nunique()), len(ea)
            db.execute("insert into events values (to_timestamp(%s),%s,%s,%s,%s,%s,"
                       "to_timestamp(%s),%s)",
                       (e.time, e.latitude, e.longitude, e.depth, n_sta, n_p, now, age))
            bus.publish("events", dict(origin=repr(e.time), lat=e.latitude, lon=e.longitude,
                                       depth=e.depth, n_sta=n_sta, n_picks=n_p,
                                       latency=f"{age:.1f}"))
            EVENTS.inc()
            LATENCY.observe(age)
            print(f"EVENT  origin {str(UTCDateTime(e.time))[:22]}  {e.latitude:.3f} "
                  f"{e.longitude:.3f}  {n_sta} sta {n_p} picks  latency {age:.0f} s", flush=True)


if __name__ == "__main__":
    main()
