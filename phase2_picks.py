#!/usr/bin/env python3
"""
Phase 2, steps 1-2 -- quake-detector

1. Download continuous waveforms for TexNet stations inside the Delaware Basin box,
   one miniSEED file per station per hour, cached under data/ (re-runs skip downloads).
2. Run pretrained PhaseNet over everything and write every pick to CSV.

Outputs:
  data/YYYYMMDD/HH/NET.STA.mseed   cached waveforms
  out/stations.csv                 station, lat, lon, elevation, band used
  out/picks.csv                    every pick: station, phase, time, probability
  timing printed at the end        download vs. inference seconds (platform metrics)

Usage:
  python phase2_picks.py                                   # default: 1 h starting 2026-10-11T07:30
  python phase2_picks.py --start 2026-10-11T00:00 --hours 6
  python phase2_picks.py --weights stead --out out_stead   # compare weights on same data
"""
import argparse
import csv
import os
import time

from obspy import UTCDateTime, Stream, read
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import FDSNNoDataException

import seisbench.models as sbm

BOX = dict(minlatitude=31.0, maxlatitude=32.2, minlongitude=-104.6, maxlongitude=-103.0)
TEXNET_URL = "http://rtserve.beg.utexas.edu"
EARTHSCOPE_URL = "https://service.earthscope.org"
BANDS = ("HH", "EH", "HN")  # preference order: broadband, short-period, strong-motion


def get_stations(client, t0, t1, nsta):
    """TX stations in the box that were operating during [t0, t1]."""
    inv = client.get_stations(network="TX", channel="HH?,EH?,HN?",
                              starttime=t0, endtime=t1, level="channel", **BOX)
    stas = []
    for net in inv:
        for sta in net:
            codes = {ch.code[:2] for ch in sta}
            band = next((b for b in BANDS if b in codes), None)
            if band:
                stas.append(dict(net=net.code, sta=sta.code, band=band,
                                 lat=sta.latitude, lon=sta.longitude,
                                 elev_m=sta.elevation))
    stas.sort(key=lambda s: s["sta"])
    if nsta and len(stas) > nsta:
        print(f"{len(stas)} stations in box; keeping first {nsta} (use --nsta 0 for all)")
        stas = stas[:nsta]
    return stas


def download_hour(s, h0, path, clients):
    """Fetch one station-hour into `path`. Returns 'cached', a server name, or None."""
    if os.path.exists(path):
        return "cached"
    for name, client in clients:
        try:
            st = client.get_waveforms(s["net"], s["sta"], "*", f"{s['band']}?", h0, h0 + 3600)
        except FDSNNoDataException:
            continue
        except Exception as e:
            print(f"    {s['sta']} {name}: {type(e).__name__}: {e}")
            continue
        loc = st[0].stats.location            # keep one location code per station
        st = st.select(location=loc)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        st.write(path, format="MSEED")
        return name
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-10-11T07:30:00")
    ap.add_argument("--hours", type=int, default=1)
    ap.add_argument("--nsta", type=int, default=0, help="max stations; 0 = all in box")
    ap.add_argument("--weights", default="instance")
    ap.add_argument("--thresh", type=float, default=0.3)
    ap.add_argument("--out", default="out")
    args = ap.parse_args()

    t0 = UTCDateTime(args.start)
    t1 = t0 + 3600 * args.hours
    os.makedirs(args.out, exist_ok=True)

    texnet = Client(base_url=TEXNET_URL)
    clients = [("TexNet", texnet), ("EarthScope", Client(base_url=EARTHSCOPE_URL))]

    # ---- stations -------------------------------------------------------
    stas = get_stations(texnet, t0, t1, args.nsta)
    with open(os.path.join(args.out, "stations.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["net", "sta", "band", "lat", "lon", "elev_m"])
        w.writeheader()
        w.writerows(stas)
    print(f"{len(stas)} stations: {' '.join(s['sta'] for s in stas)}")

    # ---- download (cached) ----------------------------------------------
    tic = time.time()
    st = Stream()
    for hr in range(args.hours):
        h0 = t0 + 3600 * hr
        hdir = os.path.join("data", h0.strftime("%Y%m%d"), h0.strftime("%H%M"))
        print(f"Hour {h0}:")
        for s in stas:
            path = os.path.join(hdir, f"{s['net']}.{s['sta']}.mseed")
            src = download_hour(s, h0, path, clients)
            print(f"  {s['sta']:6s} {s['band']}  {src or 'NO DATA'}")
            if src:
                st += read(path)
    t_download = time.time() - tic

    # Join across hour boundaries, but keep real gaps as separate segments
    # (zero-filling gaps can create false picks at the gap edges).
    st.merge(method=1, fill_value=None)
    st = st.split()
    nsamp = sum(tr.stats.npts for tr in st)
    print(f"\n{len(st)} trace segments, {nsamp / 1e6:.1f} M samples")

    # ---- inference ------------------------------------------------------
    tic = time.time()
    model = sbm.PhaseNet.from_pretrained(args.weights)
    out = model.classify(st, P_threshold=args.thresh, S_threshold=args.thresh)
    t_infer = time.time() - tic
    picks = sorted(out.picks, key=lambda p: p.peak_time)

    with open(os.path.join(args.out, "picks.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["net", "sta", "phase", "time", "prob", "start", "end"])
        for p in picks:
            net, sta = p.trace_id.split(".")[:2]
            w.writerow([net, sta, p.phase, str(p.peak_time), f"{p.peak_value:.3f}",
                        str(p.start_time), str(p.end_time)])

    # ---- summary --------------------------------------------------------
    print(f"\n{'sta':6s} {'P':>5s} {'S':>5s}")
    for s in stas:
        nP = sum(1 for p in picks if p.trace_id.split(".")[1] == s["sta"] and p.phase == "P")
        nS = sum(1 for p in picks if p.trace_id.split(".")[1] == s["sta"] and p.phase == "S")
        print(f"{s['sta']:6s} {nP:5d} {nS:5d}")
    print(f"total  {sum(p.phase == 'P' for p in picks):5d} {sum(p.phase == 'S' for p in picks):5d}")

    data_hours = len(stas) * args.hours
    print(f"\nTiming: download {t_download:.1f} s | inference {t_infer:.1f} s "
          f"for {data_hours} station-hours "
          f"({t_infer / max(data_hours, 1):.2f} s per station-hour)")
    print(f"Wrote {args.out}/stations.csv and {args.out}/picks.csv ({len(picks)} picks)")


if __name__ == "__main__":
    main()
