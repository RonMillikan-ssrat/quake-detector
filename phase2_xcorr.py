#!/usr/bin/env python3
"""
Phase 2, step 7 -- quake-detector: waveform cross-correlation check

Templates  = detected events that matched a catalog event (out/matches.csv).
Candidates = all other detected events.

For every (candidate, template) pair and every station where BOTH have a P pick:
  - cut a window around each event's own P pick (default P-0.5 s to P+4.5 s, which
    holds P and S at the near stations), bandpass 2-15 Hz
  - cross-correlate each component, allowing a small time shift (default +/-0.5 s)
  - take the signed peak (a polarity flip counts as dissimilar), average components
-> one correlation coefficient (CC) per pair per station

Baselines put the candidate numbers in context:
  template vs template : what real co-located events look like
  cross-station        : event windows from DIFFERENT stations -- impulsive signals with
                         similar envelopes but unrelated detail; the stricter null
  noise vs template    : random windows with no picks nearby (fixed seed); a loose null,
                         because noise lacks an event-like envelope

Inputs:  out/events.csv, out/assignments.csv, out/stations.csv, out/matches.csv,
         out/picks.csv, cached waveforms under data/
Outputs: out/xcorr.csv (long form: kind, event, template, station, cc)
         summary table on stdout

Usage:
  python phase2_xcorr.py
  python phase2_xcorr.py --fmin 3 --fmax 12 --win-post 3.5
"""
import argparse
import glob
import os
import random

import numpy as np
import pandas as pd
from obspy import UTCDateTime, Stream, read
from obspy.signal.cross_correlation import correlate, xcorr_max


def comp_key(ch):
    c = ch[-1]
    return "Z" if c == "Z" else ("H1" if c in "N1" else "H2")


def cut(net, sta, t0, t1, fmin, fmax):
    """Filtered 3-component window for one station. Returns (sampling_rate, {comp: array})."""
    st = Stream()
    for f in glob.glob(os.path.join("data", "*", "*", f"{net}.{sta}.mseed")):
        st += read(f, starttime=t0 - 5, endtime=t1 + 5)   # pad so taper/filter edges fall outside
    if not st:
        return None
    st.merge(method=1, fill_value=None)
    st = st.split()
    st.detrend("demean")
    st.taper(0.05)
    st.filter("bandpass", freqmin=fmin, freqmax=fmax, zerophase=True)
    st.trim(t0, t1)
    out, sr = {}, None
    for tr in st:
        need = int(round((t1 - t0) * tr.stats.sampling_rate))
        if tr.stats.npts >= need:                          # skip windows cut by data gaps
            out[comp_key(tr.stats.channel)] = tr.data[:need].astype(float)
            sr = tr.stats.sampling_rate
    return (sr, out) if out else None


def station_cc(wa, wb, shift_s):
    """Mean over shared components of the signed CC peak within +/- shift_s."""
    (sra, a), (srb, b) = wa, wb
    if sra != srb:
        return None
    shift = int(shift_s * sra)
    vals = []
    for c in a.keys() & b.keys():
        n = min(len(a[c]), len(b[c]))
        cc = correlate(a[c][:n], b[c][:n], shift)
        vals.append(xcorr_max(cc, abs_max=False)[1])
    return float(np.mean(vals)) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--start", default="2026-10-11T07:30:00")
    ap.add_argument("--hours", type=int, default=1)
    ap.add_argument("--fmin", type=float, default=2.0)
    ap.add_argument("--fmax", type=float, default=15.0)
    ap.add_argument("--win-pre", type=float, default=0.5, help="s before P pick")
    ap.add_argument("--win-post", type=float, default=4.5, help="s after P pick")
    ap.add_argument("--shift", type=float, default=0.5, help="max lag, s")
    ap.add_argument("--n-noise", type=int, default=20, help="noise windows per station")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    stations = pd.read_csv(os.path.join(args.out, "stations.csv"))
    stations["id"] = stations["net"] + "." + stations["sta"]
    sta_by_id = {r.id: (r.net, r.sta) for r in stations.itertuples()}
    events = pd.read_csv(os.path.join(args.out, "events.csv"))
    asg = pd.read_csv(os.path.join(args.out, "assignments.csv"))
    m = pd.read_csv(os.path.join(args.out, "matches.csv"))
    tmpl_mag = {int(r.detected_idx): f"M{r.usgs_mag:.1f}"
                for r in m.dropna(subset=["detected_idx"]).itertuples()}
    templates = sorted(tmpl_mag)
    candidates = [i for i in events["idx"] if i not in tmpl_mag]
    if not templates:
        raise SystemExit("No catalog-matched events to use as templates.")

    # P pick time per (event, station): highest-probability P the associator assigned
    ptime = {}
    for (ev, sid), g in asg[asg["phase"] == "P"].groupby(["event_idx", "station"]):
        ptime[(int(ev), sid)] = UTCDateTime(g.sort_values("probability").iloc[-1]["time"])

    cache = {}

    def win(ev, sid):
        if (ev, sid) not in cache:
            net, sta = sta_by_id[sid]
            tp = ptime[(ev, sid)]
            cache[(ev, sid)] = cut(net, sta, tp - args.win_pre, tp + args.win_post,
                                   args.fmin, args.fmax)
        return cache[(ev, sid)]

    def stations_of(ev):
        return {sid for (e, sid) in ptime if e == ev and sid in sta_by_id}

    rows = []

    def compare(kind, ev, tm):
        for sid in sorted(stations_of(ev) & stations_of(tm)):
            a, b = win(ev, sid), win(tm, sid)
            if a and b:
                cc = station_cc(a, b, args.shift)
                if cc is not None:
                    rows.append(dict(kind=kind, event=ev, template=tm,
                                     station=sid.split(".")[1], cc=round(cc, 3)))

    for c in candidates:
        for t in templates:
            compare("candidate", c, t)
    for i, t1 in enumerate(templates):
        for t2 in templates[i + 1:]:
            compare("template", t1, t2)

    # ---- cross-station baseline: real event windows, but DIFFERENT stations (A != B).
    # Same kind of signal (impulsive, similar envelope), unrelated waveform detail.
    # This is the stricter null: what two unrelated earthquake signals score.
    for t1 in templates:
        for t2 in templates:
            for sa in sorted(stations_of(t1)):
                for sb in sorted(stations_of(t2)):
                    if sa == sb:
                        continue
                    a, b = win(t1, sa), win(t2, sb)
                    if a and b:
                        cc = station_cc(a, b, args.shift)
                        if cc is not None:
                            rows.append(dict(kind="xsta", event=t1, template=t2,
                                             station=f"{sa.split('.')[1]}/{sb.split('.')[1]}",
                                             cc=round(cc, 3)))

    # ---- noise baseline: random windows with no pick at that station within +/-15 s
    picks = pd.read_csv(os.path.join(args.out, "picks.csv"))
    picks["ts"] = picks["time"].apply(lambda t: UTCDateTime(t).timestamp)
    t_lo = UTCDateTime(args.start).timestamp + 30
    t_hi = t_lo - 30 + 3600 * args.hours - 30
    rng = random.Random(args.seed)
    used = sorted({sid for t in templates for sid in stations_of(t)})
    for sid in used:
        net, sta = sta_by_id[sid]
        pts = picks.loc[(picks["net"] == net) & (picks["sta"] == sta), "ts"].values
        got, tries = 0, 0
        while got < args.n_noise and tries < 50 * args.n_noise:
            tries += 1
            t = rng.uniform(t_lo, t_hi)
            if len(pts) and np.min(np.abs(pts - t)) < 15:
                continue
            w = cut(net, sta, UTCDateTime(t) - args.win_pre, UTCDateTime(t) + args.win_post,
                    args.fmin, args.fmax)
            if not w:
                continue
            got += 1
            for tm in templates:
                if (tm, sid) in ptime and win(tm, sid):
                    cc = station_cc(w, win(tm, sid), args.shift)
                    if cc is not None:
                        rows.append(dict(kind="noise", event=-1, template=tm,
                                         station=sta, cc=round(cc, 3)))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "xcorr.csv"), index=False)

    # ---- report ---------------------------------------------------------
    noise = df.loc[df["kind"] == "noise", "cc"]
    p95, p99 = (noise.quantile(0.95), noise.quantile(0.99)) if len(noise) else (np.nan, np.nan)
    print(f"Window P-{args.win_pre}..P+{args.win_post} s, {args.fmin}-{args.fmax} Hz, "
          f"lag +/-{args.shift} s, components averaged, signed CC")
    print(f"\nNoise baseline ({len(noise)} station-level CCs): "
          f"median {noise.median():.2f}  95th {p95:.2f}  99th {p99:.2f}")

    xs = df.loc[df["kind"] == "xsta", "cc"]
    x95, x99 = (xs.quantile(0.95), xs.quantile(0.99)) if len(xs) else (np.nan, np.nan)
    print(f"Cross-station baseline ({len(xs)} CCs, unrelated earthquake signals): "
          f"median {xs.median():.2f}  95th {x95:.2f}  99th {x99:.2f}")

    tt = df[df["kind"] == "template"]
    if len(tt):
        g = tt.groupby(["event", "template"])["cc"].agg(["count", "mean"])
        print("\nTemplate vs template (cataloged events, what 'real' looks like):")
        for (e1, e2), r in g.iterrows():
            print(f"  {e1:>3d} {tmpl_mag[e1]:5s} vs {e2:>3d} {tmpl_mag[e2]:5s}  "
                  f"{int(r['count'])} sta  mean CC {r['mean']:.2f}")

    ev_time = dict(zip(events["idx"], events["time_utc"]))
    print(f"\n{'cand':>4s} {'time (UTC)':22s} {'best tmpl':>10s} {'sta':>3s} "
          f"{'mean':>5s} {'min':>5s} {'max':>5s}  >noise99 >xsta99")
    cand = df[df["kind"] == "candidate"]
    for c in candidates:
        cc = cand[cand["event"] == c]
        if cc.empty:
            print(f"{c:>4d} {ev_time[c][:22]:22s} {'-':>10s}   0   (no stations shared with a template)")
            continue
        g = cc.groupby("template")["cc"].agg(["count", "mean", "min", "max"])
        t = g["mean"].idxmax()
        r = g.loc[t]
        f1 = "yes" if r["mean"] > p99 else "no"
        f2 = "yes" if r["mean"] > x99 else "no"
        print(f"{c:>4d} {ev_time[c][:22]:22s} {t:>4d} {tmpl_mag[t]:>5s} {int(r['count']):3d} "
              f"{r['mean']:5.2f} {r['min']:5.2f} {r['max']:5.2f}  {f1:>7s} {f2:>7s}")
    print(f"\nWrote {args.out}/xcorr.csv")


if __name__ == "__main__":
    main()
