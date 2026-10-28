#!/usr/bin/env python3
"""
Phase 3 evaluation -- quake-detector: score a live run against the catalog

Reads a live.db written by phase3_live.py, pulls the TexNet/USGS catalog for the same
span (no magnitude floor), and matches events by origin time and epicenter distance.

Evaluation window: the run span minus the first --warmup-s (buffer filling) and the
last --settle-s (events too recent to have been reported). Catalog events outside it
are not counted as misses.

Reports:
  - catalog events: matched or missed, with magnitude and review status
  - live events: matched or not in catalog, latency
  - recall, latency statistics
  - possible split events: two live events within --split-s of each other where at
    most one matched (one earthquake reported twice by the associator)

Usage:
  python phase3_eval.py                          # live/live.db
  python phase3_eval.py --db mac_run/live.db     # another run
"""
import argparse
import sqlite3

import numpy as np
import pandas as pd
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import FDSNNoDataException
from obspy.geodetics import gps2dist_azimuth

BOX = dict(minlatitude=31.0, maxlatitude=32.2, minlongitude=-104.6, maxlongitude=-103.0)


def km(lat1, lon1, lat2, lon2):
    return gps2dist_azimuth(lat1, lon1, lat2, lon2)[0] / 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="live/live.db")
    ap.add_argument("--warmup-s", type=float, default=120)
    ap.add_argument("--settle-s", type=float, default=30)
    ap.add_argument("--match-dt", type=float, default=3.0)
    ap.add_argument("--match-km", type=float, default=15.0)
    ap.add_argument("--split-s", type=float, default=10.0)
    args = ap.parse_args()

    con = sqlite3.connect(args.db)
    ev = pd.read_sql("select * from events", con)
    span = pd.read_sql("select min(found_utc) t0, max(found_utc) t1 from picks", con).iloc[0]
    con.close()
    if span["t0"] is None:
        raise SystemExit("No picks in database.")
    run0, run1 = UTCDateTime(span["t0"]), UTCDateTime(span["t1"])
    w0, w1 = run0 + args.warmup_s, run1 - args.settle_s
    ev["t"] = ev["origin_utc"].apply(lambda s: UTCDateTime(s))
    ev = ev[(ev["t"] >= w0) & (ev["t"] <= w1)].sort_values("t").reset_index(drop=True)

    print(f"Run {str(run0)[:19]} .. {str(run1)[:19]}  "
          f"(evaluated {str(w0)[11:19]} .. {str(w1)[11:19]} UTC, {(w1 - w0) / 60:.0f} min)")

    try:
        cat = Client("USGS").get_events(starttime=w0, endtime=w1, **BOX)
    except FDSNNoDataException:
        cat = []
    crow = []
    for e in cat:
        o = e.preferred_origin() or e.origins[0]
        m = e.preferred_magnitude() or (e.magnitudes[0] if e.magnitudes else None)
        crow.append(dict(t=o.time, lat=o.latitude, lon=o.longitude,
                         mag=m.mag if m else np.nan, mtype=m.magnitude_type if m else "",
                         mode=o.evaluation_mode or "?",
                         id=str(e.resource_id).split("eventid=")[-1].split("&")[0]))
    crow.sort(key=lambda r: r["t"])

    # one-to-one matching, closest origin time first
    pairs = []
    for ci, c in enumerate(crow):
        for li, l in ev.iterrows():
            dt = abs(l["t"] - c["t"])
            if dt <= args.match_dt:
                d = km(c["lat"], c["lon"], l["lat"], l["lon"])
                if d <= args.match_km:
                    pairs.append((dt, ci, li, d))
    pairs.sort()
    c2l, l2c = {}, {}
    for dt, ci, li, d in pairs:
        if ci not in c2l and li not in l2c:
            c2l[ci] = (li, dt, d)
            l2c[li] = ci

    print(f"\nCatalog events in window: {len(crow)}  "
          f"(reviewed {sum(c['mode'] == 'manual' for c in crow)}, "
          f"automatic {sum(c['mode'] == 'automatic' for c in crow)})")
    for ci, c in enumerate(crow):
        if ci in c2l:
            li, dt, d = c2l[ci]
            st = f"matched  live latency {ev.loc[li, 'latency_s']:.0f} s  (dt {dt:.1f} s, {d:.1f} km)"
        else:
            st = "MISSED"
        print(f"  {str(c['t'])[:22]}  M{c['mag']:.1f} {c['mtype']:3s} {c['mode']:9s} {c['id']:14s} {st}")

    print(f"\nLive events in window: {len(ev)}")
    for li, l in ev.iterrows():
        tag = (f"catalog M{crow[l2c[li]]['mag']:.1f}" if li in l2c else "not in catalog")
        print(f"  {str(l['t'])[:22]}  {l['lat']:.3f} {l['lon']:.3f}  {l['n_sta']:2d} sta  "
              f"latency {l['latency_s']:5.0f} s  {tag}")

    # possible splits: neighbouring live events close in time, not both matched
    print("\nPossible split events (one earthquake reported twice):")
    found = False
    for i in range(len(ev) - 1):
        a, b = ev.iloc[i], ev.iloc[i + 1]
        if b["t"] - a["t"] <= args.split_s and not (i in l2c and (i + 1) in l2c):
            found = True
            print(f"  {str(a['t'])[11:22]} & {str(b['t'])[11:22]}  "
                  f"{b['t'] - a['t']:.1f} s apart, {km(a['lat'], a['lon'], b['lat'], b['lon']):.0f} km apart, "
                  f"matched: {'yes' if i in l2c else 'no'} / {'yes' if (i + 1) in l2c else 'no'}")
    if not found:
        print("  none")

    lat = ev["latency_s"]
    n_m, n_c, n_l = len(c2l), len(crow), len(ev)
    print(f"\nSummary: catalog {n_c} | live {n_l} | matched {n_m} | missed {n_c - n_m} | "
          f"live-only {n_l - n_m}")
    if n_c:
        print(f"Recall vs catalog: {n_m}/{n_c} = {n_m / n_c:.0%}")
    if n_l:
        print(f"Latency: median {lat.median():.0f} s  90th pct {lat.quantile(0.9):.0f} s  "
              f"max {lat.max():.0f} s")
    if any(c["mode"] != "manual" for c in crow):
        print("Note: some catalog events are not yet analyst-reviewed; rerun later for final numbers.")


if __name__ == "__main__":
    main()
