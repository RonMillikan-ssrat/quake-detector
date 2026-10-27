#!/usr/bin/env python3
"""
Phase 2, steps 3-5 -- quake-detector

Associate PhaseNet picks into earthquakes with PyOcto, locate them with a simple
homogeneous velocity model, and score the result against the USGS catalog.

Inputs (from phase2_picks.py):
  out/stations.csv, out/picks.csv

Outputs:
  out/events.csv        one row per detected event: time, lat, lon, depth, pick/station counts
  out/assignments.csv   which pick belongs to which event, with travel-time residual
  out/matches.csv       USGS catalog events and the detected event each matched (if any)

Usage:
  python phase2_events.py
  python phase2_events.py --vp 5.8 --vs 3.3 --tol 1.5     # try other velocity assumptions
  python phase2_events.py --n-picks 8 --n-ps 4             # stricter event definition
"""
import argparse
import os

import pandas as pd
import pyocto
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import FDSNNoDataException
from obspy.geodetics import gps2dist_azimuth

BOX = dict(minlatitude=31.0, maxlatitude=32.2, minlongitude=-104.6, maxlongitude=-103.0)


def load(outdir):
    """Read phase2_picks.py output into the DataFrame shapes PyOcto expects."""
    stations = pd.read_csv(os.path.join(outdir, "stations.csv"))
    stations["id"] = stations["net"] + "." + stations["sta"]
    stations = stations.rename(columns={"lat": "latitude", "lon": "longitude",
                                        "elev_m": "elevation"})

    picks = pd.read_csv(os.path.join(outdir, "picks.csv"))
    picks["station"] = picks["net"] + "." + picks["sta"]   # must equal a station id
    picks["time"] = picks["time"].apply(lambda t: UTCDateTime(t).timestamp)
    picks = picks.rename(columns={"prob": "probability"})
    return stations, picks[["station", "time", "phase", "probability"]]


def usgs_catalog(t0, t1, minmag):
    try:
        cat = Client("USGS").get_events(starttime=t0, endtime=t1, minmagnitude=minmag, **BOX)
    except FDSNNoDataException:
        return []
    rows = []
    for ev in cat:
        o = ev.preferred_origin() or ev.origins[0]
        m = ev.preferred_magnitude() or (ev.magnitudes[0] if ev.magnitudes else None)
        rows.append(dict(time=o.time, lat=o.latitude, lon=o.longitude,
                         depth_km=(o.depth or 0) / 1000,
                         mag=m.mag if m else float("nan"),
                         magtype=m.magnitude_type if m else ""))
    return sorted(rows, key=lambda r: r["time"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--start", default="2026-10-11T07:30:00")
    ap.add_argument("--hours", type=int, default=1)
    # velocity model: one P and one S speed for the whole region (crude; 1-D model later)
    ap.add_argument("--vp", type=float, default=5.5, help="P velocity, km/s")
    ap.add_argument("--vs", type=float, default=3.1, help="S velocity, km/s")
    ap.add_argument("--tol", type=float, default=1.0, help="travel-time tolerance, s")
    ap.add_argument("--zmax", type=float, default=20.0, help="max depth, km")
    ap.add_argument("--time-before", type=float, default=80.0,
                    help="s; must exceed the longest S travel time across the box")
    # what counts as an event
    ap.add_argument("--n-picks", type=int, default=6, help="min total picks")
    ap.add_argument("--n-ps", type=int, default=3,
                    help="min stations with both P and S picks")
    # catalog matching
    ap.add_argument("--minmag", type=float, default=0.0)
    ap.add_argument("--match-dt", type=float, default=3.0, help="origin-time window, s")
    ap.add_argument("--match-km", type=float, default=15.0, help="epicenter distance, km")
    args = ap.parse_args()

    stations, picks = load(args.out)
    print(f"{len(picks)} picks on {picks['station'].nunique()} stations "
          f"({len(stations)} stations listed)")

    # ---- association ----------------------------------------------------
    vm = pyocto.VelocityModel0D(p_velocity=args.vp, s_velocity=args.vs, tolerance=args.tol)
    assoc = pyocto.OctoAssociator.from_area(
        lat=(BOX["minlatitude"], BOX["maxlatitude"]),
        lon=(BOX["minlongitude"], BOX["maxlongitude"]),
        zlim=(0.0, args.zmax),
        velocity_model=vm,
        time_before=args.time_before,
        n_picks=args.n_picks,
        n_p_and_s_picks=args.n_ps,
    )
    assoc.transform_stations(stations)            # adds x, y, z (km, local projection)
    events, assignments = assoc.associate(picks, stations)

    if len(events) == 0:
        print("No events associated. Try --n-picks 4 --n-ps 2 or a larger --tol.")
        return
    assoc.transform_events(events)                # adds latitude, longitude, depth

    per_ev = assignments.groupby("event_idx").agg(
        n_sta=("station", "nunique"),
        n_P=("phase", lambda s: (s == "P").sum()),
        n_S=("phase", lambda s: (s == "S").sum()),
        rms_res=("residual", lambda r: (r ** 2).mean() ** 0.5),
    )
    events = events.merge(per_ev, left_on="idx", right_index=True)
    events["time_utc"] = events["time"].apply(lambda t: str(UTCDateTime(t)))
    events = events.sort_values("time").reset_index(drop=True)

    events[["idx", "time_utc", "latitude", "longitude", "depth", "picks",
            "n_sta", "n_P", "n_S", "rms_res"]].to_csv(
        os.path.join(args.out, "events.csv"), index=False, float_format="%.4f")
    assignments.to_csv(os.path.join(args.out, "assignments.csv"), index=False)

    # ---- catalog comparison ---------------------------------------------
    t0 = UTCDateTime(args.start)
    t1 = t0 + 3600 * args.hours
    cat = usgs_catalog(t0, t1, args.minmag)

    matched_idx = set()
    match_rows = []
    for c in cat:
        best = None
        for _, e in events.iterrows():
            dt = abs(UTCDateTime(e["time"]) - c["time"])
            if dt > args.match_dt:
                continue
            km = gps2dist_azimuth(c["lat"], c["lon"], e["latitude"], e["longitude"])[0] / 1000
            if km <= args.match_km and (best is None or dt < best[1]):
                best = (e["idx"], dt, km)
        if best:
            matched_idx.add(best[0])
        match_rows.append(dict(usgs_time=str(c["time"]), usgs_mag=c["mag"],
                               usgs_magtype=c["magtype"], usgs_lat=c["lat"],
                               usgs_lon=c["lon"], usgs_depth_km=c["depth_km"],
                               detected_idx=best[0] if best else "",
                               dt_s=round(best[1], 2) if best else "",
                               dist_km=round(best[2], 1) if best else ""))
    pd.DataFrame(match_rows).to_csv(os.path.join(args.out, "matches.csv"), index=False)

    # ---- report ---------------------------------------------------------
    print(f"\nVelocity model: Vp={args.vp} Vs={args.vs} km/s, tolerance {args.tol} s; "
          f"event = >= {args.n_picks} picks and >= {args.n_ps} stations with P+S")
    print(f"\n{'time (UTC)':23s} {'lat':>7s} {'lon':>9s} {'dep':>5s} "
          f"{'sta':>3s} {'P':>3s} {'S':>3s} {'rms':>5s}  USGS")
    for _, e in events.iterrows():
        tag = ""
        for m in match_rows:
            if m["detected_idx"] == e["idx"]:
                tag = f"M{m['usgs_mag']:.1f} {m['usgs_magtype']} (dt {m['dt_s']} s, {m['dist_km']} km)"
        print(f"{e['time_utc'][:22]:23s} {e['latitude']:7.3f} {e['longitude']:9.3f} "
              f"{e['depth']:5.1f} {e['n_sta']:3d} {e['n_P']:3d} {e['n_S']:3d} "
              f"{e['rms_res']:5.2f}  {tag}")

    print(f"\nUSGS catalog events in window/box: {len(cat)}")
    for m in match_rows:
        status = f"matched -> event {m['detected_idx']}" if m["detected_idx"] != "" else "MISSED"
        print(f"  {m['usgs_time'][:22]}  M{m['usgs_mag']:.1f} {m['usgs_magtype']:3s}  {status}")

    n_det, n_cat, n_match = len(events), len(cat), len(matched_idx)
    print(f"\nDetected {n_det} | catalog {n_cat} | matched {n_match} | "
          f"missed {n_cat - n_match} | detected-not-in-USGS {n_det - n_match}")
    if n_cat:
        print(f"Recall vs USGS: {n_match}/{n_cat} = {n_match / n_cat:.0%}")
    print(f"Wrote {args.out}/events.csv, assignments.csv, matches.csv")


if __name__ == "__main__":
    main()
