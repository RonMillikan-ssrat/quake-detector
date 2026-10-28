#!/usr/bin/env python3
"""
Phase 3 -- quake-detector: live detection from TexNet's SeedLink stream

One process, three parts:
  ingest    a background thread subscribes to TexNet SeedLink and appends every
            arriving miniSEED record to a rolling in-memory buffer (last --keep-s s)
  picker    every --step-s seconds, PhaseNet runs on the whole buffer; only picks
            away from the buffer edges are kept, and repeats of picks already held
            are dropped (the buffer overlaps itself step to step)
  associator PyOcto runs on the last --assoc-s seconds of picks; an event is reported
            once its origin is older than --settle-s (time for distant stations' S
            waves to arrive and be picked), and only if not already reported

Detection latency = report time - origin time. It includes --settle-s by design.

Outputs:
  stdout          one status line per cycle, one EVENT line per detection
  live/live.db    SQLite: events and picks tables

Usage:
  python phase3_live.py                    # run until Ctrl-C
  python phase3_live.py --minutes 30       # timed run
"""
import argparse
import os
import sqlite3
import threading
import time

import numpy as np
import pandas as pd
import pyocto
from obspy import UTCDateTime, Stream
from obspy.clients.fdsn import Client as FDSNClient
from obspy.clients.seedlink.basic_client import Client as SLInfoClient
from obspy.clients.seedlink.easyseedlink import EasySeedLinkClient
from obspy.geodetics import gps2dist_azimuth

import seisbench.models as sbm

BOX = dict(minlatitude=31.0, maxlatitude=32.2, minlongitude=-104.6, maxlongitude=-103.0)
SL_HOST, SL_PORT = "rtserve.beg.utexas.edu", 18000
FDSN_URL = "http://rtserve.beg.utexas.edu"
BANDS = ("HH", "EH", "HN")


# ---------------------------------------------------------------- ingest
class Buffer:
    """Thread-safe rolling waveform buffer shared by ingest and processing."""

    def __init__(self, keep_s):
        self.keep_s = keep_s
        self.lock = threading.Lock()
        self.st = Stream()
        self.n_records = 0

    def add(self, tr):
        with self.lock:
            self.st += tr
            self.n_records += 1

    def snapshot(self):
        """Merge, drop data older than keep_s, return an independent copy."""
        with self.lock:
            if not self.st:
                return Stream()
            self.st.merge(method=1, fill_value=None)
            end = max(tr.stats.endtime for tr in self.st)
            self.st.trim(end - self.keep_s, None)
            for tr in self.st:          # ObsPy logs every merge/trim; it caps at 100 and warns
                tr.stats.processing = []
            return self.st.copy()


class SeedLinkFeed(EasySeedLinkClient):
    def __init__(self, url, buf, connect_timeout=30):
        super().__init__(url, autoconnect=False)
        self.buf = buf
        # ObsPy 1.5 workaround. SeedLinkConnection.connect() crashes when timeout is None
        # (compares elapsed time to None). But the same `timeout` attribute also caps how
        # long collect() streams, so it must NOT stay set. Set it only around connect(),
        # including the reconnects the library does internally after network timeouts.
        conn = self.conn
        orig_connect = conn.connect

        def connect_with_timeout():
            conn.timeout = connect_timeout
            try:
                return orig_connect()
            finally:
                conn.timeout = None

        conn.connect = connect_with_timeout
        self.connect()

    def on_data(self, trace):          # called once per arriving miniSEED record
        self.buf.add(trace)

    def on_seedlink_error(self):
        print("SeedLink error reported by server", flush=True)


def start_feed(stations, buf):
    feed = SeedLinkFeed(f"{SL_HOST}:{SL_PORT}", buf)
    for s in stations:
        feed.select_stream(s["net"], s["sta"], f"{s['band']}?")
    th = threading.Thread(target=feed.run, daemon=True)
    th.start()
    return th


# ---------------------------------------------------------------- stations
def get_stations():
    now = UTCDateTime()
    inv = FDSNClient(base_url=FDSN_URL).get_stations(
        network="TX", channel="HH?,EH?,HN?", starttime=now - 86400, endtime=now,
        level="channel", **BOX)
    streamed = {(n, s) for n, s in SLInfoClient(SL_HOST, SL_PORT, timeout=30)
                .get_info(network="TX", level="station")}
    stas = []
    for net in inv:
        for sta in net:
            codes = {ch.code[:2] for ch in sta}
            band = next((b for b in BANDS if b in codes), None)
            if band and (net.code, sta.code) in streamed:
                stas.append(dict(net=net.code, sta=sta.code, band=band, lat=sta.latitude,
                                 lon=sta.longitude, elev_m=sta.elevation))
    return sorted(stas, key=lambda s: s["sta"])


# ---------------------------------------------------------------- main loop
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=0, help="0 = run until Ctrl-C")
    ap.add_argument("--weights", default="instance")
    ap.add_argument("--thresh", type=float, default=0.3)
    ap.add_argument("--keep-s", type=float, default=90, help="waveform buffer length")
    ap.add_argument("--step-s", type=float, default=10, help="seconds between cycles")
    ap.add_argument("--edge-s", type=float, default=5,
                    help="ignore picks this close to either buffer edge")
    ap.add_argument("--dedup-s", type=float, default=0.5,
                    help="same station+phase within this = same pick")
    ap.add_argument("--assoc-s", type=float, default=300, help="pick window for association")
    ap.add_argument("--settle-s", type=float, default=30,
                    help="wait this long after origin before reporting")
    ap.add_argument("--vp", type=float, default=5.5)
    ap.add_argument("--vs", type=float, default=3.1)
    ap.add_argument("--tol", type=float, default=1.0)
    ap.add_argument("--n-picks", type=int, default=6)
    ap.add_argument("--n-ps", type=int, default=3)
    ap.add_argument("--outdir", default="live")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    db = sqlite3.connect(os.path.join(args.outdir, "live.db"))
    db.execute("create table if not exists events (origin_utc text, lat real, lon real, "
               "depth_km real, n_sta int, n_picks int, reported_utc text, latency_s real)")
    db.execute("create table if not exists picks (station text, phase text, time_utc text, "
               "prob real, found_utc text)")

    stations = get_stations()
    print(f"{len(stations)} Delaware Basin stations in the live stream: "
          f"{' '.join(s['sta'] for s in stations)}", flush=True)
    sdf = pd.DataFrame(stations)
    sdf["id"] = sdf["net"] + "." + sdf["sta"]
    sdf = sdf.rename(columns={"lat": "latitude", "lon": "longitude", "elev_m": "elevation"})

    assoc = pyocto.OctoAssociator.from_area(
        lat=(BOX["minlatitude"], BOX["maxlatitude"]),
        lon=(BOX["minlongitude"], BOX["maxlongitude"]),
        zlim=(0.0, 20.0),
        velocity_model=pyocto.VelocityModel0D(p_velocity=args.vp, s_velocity=args.vs,
                                              tolerance=args.tol),
        time_before=80.0, n_picks=args.n_picks, n_p_and_s_picks=args.n_ps)
    assoc.transform_stations(sdf)

    model = sbm.PhaseNet.from_pretrained(args.weights)
    buf = Buffer(args.keep_s)
    feed_thread = start_feed(stations, buf)
    print(f"Connected to {SL_HOST}:{SL_PORT}; filling {args.keep_s:.0f} s buffer...", flush=True)

    picks, reported = [], []
    t_stop = time.time() + 60 * args.minutes if args.minutes else None
    try:
        while t_stop is None or time.time() < t_stop:
            cycle_start = time.time()
            if not feed_thread.is_alive():
                print("SeedLink thread stopped; reconnecting", flush=True)
                feed_thread = start_feed(stations, buf)

            st = buf.snapshot().split()           # gaps -> separate segments
            st = Stream([tr for tr in st if tr.stats.endtime - tr.stats.starttime >= 35])
            now = UTCDateTime()
            if not st:
                time.sleep(args.step_s)
                continue

            ends, starts = {}, {}
            for tr in st:
                sid = f"{tr.stats.network}.{tr.stats.station}"
                ends[sid] = max(ends.get(sid, tr.stats.endtime), tr.stats.endtime)
                starts[sid] = min(starts.get(sid, tr.stats.starttime), tr.stats.starttime)

            # ---- pick
            tic = time.time()
            out = model.classify(st, P_threshold=args.thresh, S_threshold=args.thresh)
            t_inf = time.time() - tic
            n_new = 0
            for p in out.picks:
                sid = ".".join(p.trace_id.split(".")[:2])
                if not (starts[sid] + args.edge_s < p.peak_time < ends[sid] - args.edge_s):
                    continue                     # near an edge: wait for a later cycle
                ts = p.peak_time.timestamp
                if any(q["station"] == sid and q["phase"] == p.phase
                       and abs(q["time"] - ts) < args.dedup_s for q in picks):
                    continue                     # already held from an earlier cycle
                picks.append(dict(station=sid, phase=p.phase, time=ts,
                                  probability=float(p.peak_value)))
                db.execute("insert into picks values (?,?,?,?,?)",
                           (sid, p.phase, str(p.peak_time), float(p.peak_value), str(now)))
                n_new += 1
            picks = [q for q in picks if q["time"] > now.timestamp - 2 * args.assoc_s]

            # ---- associate and report
            recent = [q for q in picks if q["time"] > now.timestamp - args.assoc_s]
            tic = time.time()
            if len(recent) >= args.n_picks:
                events, asg = assoc.associate(
                    pd.DataFrame(recent)[["station", "time", "phase", "probability"]], sdf)
                if len(events):
                    assoc.transform_events(events)
                    for e in events.itertuples():
                        age = now.timestamp - e.time
                        # too new: wait to settle; too old: its picks may be partly gone
                        if age < args.settle_s or age > args.assoc_s - 90:
                            continue
                        if any(abs(e.time - r["time"]) < 3 and
                               gps2dist_azimuth(e.latitude, e.longitude,
                                                r["lat"], r["lon"])[0] < 15000
                               for r in reported):
                            continue
                        ea = asg[asg["event_idx"] == e.idx]
                        rec = dict(time=e.time, lat=e.latitude, lon=e.longitude)
                        reported.append(rec)
                        origin = UTCDateTime(e.time)
                        db.execute("insert into events values (?,?,?,?,?,?,?,?)",
                                   (str(origin), e.latitude, e.longitude, e.depth,
                                    int(ea["station"].nunique()), len(ea), str(now), age))
                        print(f"EVENT  origin {str(origin)[:22]}  {e.latitude:.3f} "
                              f"{e.longitude:.3f}  {ea['station'].nunique()} sta "
                              f"{len(ea)} picks  latency {age:.0f} s", flush=True)
            t_assoc = time.time() - tic
            db.commit()

            # ---- status
            lat = [now - t for t in ends.values()]
            cycle = time.time() - cycle_start
            print(f"[{str(now)[11:19]}] {len(ends)}/{len(stations)} sta  "
                  f"data lag med {np.median(lat):.1f} s max {max(lat):.1f} s  "
                  f"infer {t_inf:.2f} s  assoc {t_assoc:.2f} s  cycle {cycle:.1f} s  "
                  f"+{n_new} picks ({len(picks)} held)  "
                  f"{len(reported)} events", flush=True)

            time.sleep(max(0.0, args.step_s - (time.time() - cycle_start)))
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        db.commit()
        db.close()
        print(f"{len(reported)} events reported; see {args.outdir}/live.db")


if __name__ == "__main__":
    main()
