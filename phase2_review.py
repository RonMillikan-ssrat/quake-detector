#!/usr/bin/env python3
"""
Phase 2, step 6 -- quake-detector: visual review of detected events

One figure per detected event: a record section (stations sorted by distance from
the detected hypocenter), vertical component on the left, a horizontal on the right.
  solid blue / red   = PhaseNet P / S picks the associator assigned to this event
  dotted blue / red  = P / S arrival predicted by the homogeneous velocity model
  grey station label = station within --extra-km that has NO pick for this event
                       (a quiet near station is evidence against a real event)

Inputs:  out/events.csv, out/assignments.csv, out/stations.csv, out/matches.csv,
         cached waveforms under data/
Outputs: review/event_NN.png

Usage:
  python phase2_review.py                 # all events
  python phase2_review.py --only-new      # only events not in the catalog
  python phase2_review.py --events 3 5 9  # specific event idx values
"""
import argparse
import glob
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from obspy import UTCDateTime, Stream, read
from obspy.geodetics import gps2dist_azimuth


def window(net, sta, t0, t1):
    """Waveforms for one station between t0 and t1 from the local cache."""
    st = Stream()
    for f in glob.glob(os.path.join("data", "*", "*", f"{net}.{sta}.mseed")):
        st += read(f, starttime=t0, endtime=t1)
    if not st:
        return st
    st.merge(method=1, fill_value=None)
    st = st.split()                      # gaps -> separate segments (filters can't take masks)
    st.detrend("demean")
    st.taper(0.05)
    st.filter("bandpass", freqmin=2.0, freqmax=20.0)
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--vp", type=float, default=5.5)
    ap.add_argument("--vs", type=float, default=3.1)
    ap.add_argument("--pre", type=float, default=3.0, help="s before origin")
    ap.add_argument("--post", type=float, default=20.0, help="s after origin")
    ap.add_argument("--extra-km", type=float, default=25.0,
                    help="also show unpicked stations within this hypocentral distance")
    ap.add_argument("--only-new", action="store_true")
    ap.add_argument("--events", type=int, nargs="*")
    args = ap.parse_args()

    stations = pd.read_csv(os.path.join(args.out, "stations.csv"))
    stations["id"] = stations["net"] + "." + stations["sta"]
    events = pd.read_csv(os.path.join(args.out, "events.csv"))
    asg = pd.read_csv(os.path.join(args.out, "assignments.csv"))

    catalog = {}
    mpath = os.path.join(args.out, "matches.csv")
    if os.path.exists(mpath):
        m = pd.read_csv(mpath)
        if "detected_idx" in m.columns:
            for r in m.dropna(subset=["detected_idx"]).itertuples():
                catalog[int(r.detected_idx)] = f"catalog M{r.usgs_mag:.1f} {r.usgs_magtype}"

    if args.events:
        events = events[events["idx"].isin(args.events)]
    if args.only_new:
        events = events[~events["idx"].isin(catalog.keys())]

    os.makedirs("review", exist_ok=True)
    for e in events.itertuples():
        t0 = UTCDateTime(e.time_utc)
        ep = asg[asg["event_idx"] == e.idx]
        picked = set(ep["station"])

        rows = []
        for s in stations.itertuples():
            epi = gps2dist_azimuth(e.latitude, e.longitude, s.lat, s.lon)[0] / 1000
            hyp = math.hypot(epi, e.depth + s.elev_m / 1000)
            if s.id in picked or hyp <= args.extra_km:
                rows.append((hyp, s))
        rows.sort(key=lambda r: r[0])

        fig, (axz, axh) = plt.subplots(1, 2, figsize=(14, 0.6 * len(rows) + 1.8), sharey=True)
        for i, (hyp, s) in enumerate(rows):
            st = window(s.net, s.sta, t0 - args.pre, t0 + args.post)
            for ax, comps in ((axz, "Z"), (axh, "N1")):
                trs = [tr for tr in st if tr.stats.channel[-1] in comps]
                if trs:
                    norm = max(abs(tr.data).max() for tr in trs) or 1.0
                    for tr in trs:
                        ax.plot(tr.times(reftime=t0), i + 0.45 * tr.data / norm, "k", lw=0.4)
                for v, c in ((args.vp, "b"), (args.vs, "r")):
                    tp = hyp / v
                    ax.plot([tp, tp], [i - 0.45, i + 0.45], c + ":", lw=1)
                for p in ep[ep["station"] == s.id].itertuples():
                    tp = p.time - t0.timestamp
                    ax.plot([tp, tp], [i - 0.45, i + 0.45],
                            "b" if p.phase == "P" else "r", lw=1.6)

        labels = [f"{s.sta} {hyp:4.0f} km" for hyp, s in rows]
        axz.set_yticks(range(len(rows)))
        axz.set_yticklabels(labels, fontsize=8)
        for lab, (hyp, s) in zip(axz.get_yticklabels(), rows):
            if s.id not in picked:
                lab.set_color("0.6")
        axz.invert_yaxis()
        for ax, name in ((axz, "vertical"), (axh, "horizontal (N/1)")):
            ax.set_xlim(-args.pre, args.post)
            ax.set_xlabel(f"s after detected origin ({name})")
            ax.axvline(0, color="k", ls="--", lw=0.8)

        tag = catalog.get(e.idx, "NOT IN CATALOG")
        fig.suptitle(f"Event {e.idx}   {e.time_utc[:22]}   {e.latitude:.3f}, {e.longitude:.3f}   "
                     f"{e.n_sta} sta  P{e.n_P} S{e.n_S}  rms {e.rms_res:.2f} s   [{tag}]",
                     fontsize=10)
        fig.tight_layout()
        path = os.path.join("review", f"event_{e.idx:02d}.png")
        fig.savefig(path, dpi=110)
        plt.close(fig)
        print(f"{path}  {tag}")


if __name__ == "__main__":
    main()
