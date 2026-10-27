#!/usr/bin/env python3
"""
Phase 1 -- quake-detector

Find the largest recent USGS-cataloged earthquake in the Delaware Basin (West Texas),
pull TexNet waveforms from the nearest stations, run a pretrained SeisBench PhaseNet
picker, and compare picks against the catalog origin time and a TauP travel-time
prediction.

Outputs:
  - a per-station table printed to stdout (distance, picked P/S, predicted P/S, residuals)
  - phase1_picks.png  (Z components with picks, origin time, predicted arrivals)

Usage:
  python phase1_picks.py --list-weights          # show available PhaseNet weights
  python phase1_picks.py                         # defaults: last 30 days, M>=2.5, 6 stations
  python phase1_picks.py --weights instance --days 60 --nsta 8
"""
import argparse
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from obspy import UTCDateTime, Stream
from obspy.clients.fdsn import Client
from obspy.geodetics import locations2degrees, degrees2kilometers
from obspy.taup import TauPyModel

import seisbench.models as sbm

# Approximate Delaware Basin bounding box (Pecos / Reeves / Culberson area)
BOX = dict(minlatitude=31.0, maxlatitude=32.2, minlongitude=-104.6, maxlongitude=-103.0)

TEXNET_URL = "http://rtserve.beg.utexas.edu"       # TexNet FDSN (dataselect + station)
EARTHSCOPE_URL = "https://service.earthscope.org"   # fallback for waveforms


def find_event(days, minmag):
    """Largest USGS ComCat event in the box over the last `days` days."""
    usgs = Client("USGS")
    end = UTCDateTime()
    cat = usgs.get_events(starttime=end - days * 86400, endtime=end,
                          minmagnitude=minmag, orderby="magnitude", limit=1, **BOX)
    if len(cat) == 0:
        sys.exit(f"No M>={minmag} events in box in last {days} days; try --days or --minmag")
    ev = cat[0]
    o = ev.preferred_origin() or ev.origins[0]
    m = ev.preferred_magnitude() or ev.magnitudes[0]
    depth_km = (o.depth or 5000.0) / 1000.0
    print(f"Event: {o.time}  lat={o.latitude:.3f} lon={o.longitude:.3f} "
          f"depth={depth_km:.1f} km  {m.magnitude_type}{m.mag:.1f}")
    return o.time, o.latitude, o.longitude, depth_km


def nearest_stations(client, t0, lat, lon, nsta, radius_deg):
    inv = client.get_stations(network="TX", channel="HH?,EH?,HN?",
                              starttime=t0, endtime=t0 + 60,
                              latitude=lat, longitude=lon, maxradius=radius_deg,
                              level="channel")
    stas = []
    for net in inv:
        for sta in net:
            d_km = degrees2kilometers(locations2degrees(lat, lon, sta.latitude, sta.longitude))
            # prefer broadband HH, then EH, then strong-motion HN
            codes = {ch.code[:2] for ch in sta}
            band = next((b for b in ("HH", "EH", "HN") if b in codes), None)
            if band:
                stas.append((d_km, net.code, sta.code, band))
    stas.sort()
    return stas[:nsta]


def fetch(t0, stas, pre, post):
    st = Stream()
    for d_km, net, sta, band in stas:
        for name, url in (("TexNet", TEXNET_URL), ("EarthScope", EARTHSCOPE_URL)):
            try:
                tr = Client(base_url=url).get_waveforms(net, sta, "*", f"{band}?",
                                                        t0 - pre, t0 + post)
                # keep a single location code per station
                loc = tr[0].stats.location
                st += tr.select(location=loc)
                print(f"  {net}.{sta:6s} {band}  {d_km:6.1f} km  <- {name}")
                break
            except Exception as e:
                print(f"  {net}.{sta:6s} {band}  {name} failed: {type(e).__name__}")
    st.merge(method=1, fill_value="interpolate")
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-weights", action="store_true")
    ap.add_argument("--weights", default="instance")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--minmag", type=float, default=2.5)
    ap.add_argument("--nsta", type=int, default=6)
    ap.add_argument("--radius", type=float, default=1.0, help="station search radius, degrees")
    ap.add_argument("--pre", type=float, default=60.0, help="seconds before origin")
    ap.add_argument("--post", type=float, default=120.0, help="seconds after origin")
    ap.add_argument("--thresh", type=float, default=0.3)
    args = ap.parse_args()

    if args.list_weights:
        print(sbm.PhaseNet.list_pretrained())
        return

    t0, lat, lon, depth_km = find_event(args.days, args.minmag)

    print("Stations:")
    stas = nearest_stations(Client(base_url=TEXNET_URL), t0, lat, lon, args.nsta, args.radius)
    if not stas:
        sys.exit("No TX stations found; try --radius 2")
    st = fetch(t0, stas, args.pre, args.post)
    print(st.__str__(extended=True))

    print(f"Loading PhaseNet weights '{args.weights}'")
    model = sbm.PhaseNet.from_pretrained(args.weights)
    out = model.classify(st, P_threshold=args.thresh, S_threshold=args.thresh)
    picks = out.picks
    print(f"{len(picks)} picks")
    print(f"\n{'sta':6s} {'phase':5s} {'t_after_origin':>14s} {'prob':>5s}")
    for p in sorted(picks, key=lambda p: (p.trace_id, p.peak_time)):
        print(f"{p.trace_id.split('.')[1]:6s} {p.phase:5s} {p.peak_time - t0:14.2f} "
              f"{p.peak_value:5.2f}")

    def nearest(sta, phase, pred):
        """Pick of this phase at this station closest in time to the predicted arrival."""
        cands = [p for p in picks
                 if p.trace_id.split(".")[1] == sta and p.phase == phase]
        return min(cands, key=lambda p: abs((p.peak_time - t0) - pred)) if cands else None

    taup = TauPyModel("iasp91")  # global model: a sanity check, not a local velocity model
    print(f"\n{'sta':6s} {'dist_km':>7s} {'P_pick':>7s} {'P_pred':>7s} {'P_res':>6s} "
          f"{'S_pick':>7s} {'S_pred':>7s} {'S_res':>6s}")
    rows = []
    for d_km, net, sta, band in stas:
        arr = taup.get_travel_times(source_depth_in_km=depth_km,
                                    distance_in_degree=d_km / 111.19,
                                    phase_list=["p", "P", "s", "S"])
        p_pred = min((a.time for a in arr if a.name.upper() == "P"), default=float("nan"))
        s_pred = min((a.time for a in arr if a.name.upper() == "S"), default=float("nan"))
        P = nearest(sta, "P", p_pred)
        S = nearest(sta, "S", s_pred)
        p_obs = (P.peak_time - t0) if P else float("nan")
        s_obs = (S.peak_time - t0) if S else float("nan")
        print(f"{sta:6s} {d_km:7.1f} {p_obs:7.2f} {p_pred:7.2f} {p_obs - p_pred:6.2f} "
              f"{s_obs:7.2f} {s_pred:7.2f} {s_obs - s_pred:6.2f}")
        rows.append((d_km, sta, p_pred, s_pred))

    # Plot: one panel per station, Z component, bandpassed for display only
    disp = st.copy().detrend("demean").filter("bandpass", freqmin=1, freqmax=20)
    fig, axes = plt.subplots(len(rows), 1, figsize=(11, 1.8 * len(rows)), sharex=True)
    if len(rows) == 1:
        axes = [axes]
    for ax, (d_km, sta, p_pred, s_pred) in zip(axes, rows):
        z = disp.select(station=sta, component="Z")
        if z:
            tr = z[0]
            ax.plot(tr.times(reftime=t0), tr.data, "k", lw=0.5)
        for p in picks:
            if p.trace_id.split(".")[1] == sta:
                ax.axvline(p.peak_time - t0, color="b" if p.phase == "P" else "r", lw=1.5)
        ax.axvline(0, color="k", ls="--", lw=1)
        ax.axvline(p_pred, color="b", ls=":", lw=1)
        ax.axvline(s_pred, color="r", ls=":", lw=1)
        ax.set_ylabel(f"{sta}\n{d_km:.0f} km", rotation=0, ha="right", va="center")
        ax.set_yticks([])
    axes[-1].set_xlabel("seconds after catalog origin")
    axes[0].set_title("PhaseNet picks (solid: blue=P, red=S) | origin (dashed) | "
                      "iasp91 predicted (dotted)")
    plt.tight_layout()
    plt.savefig("phase1_picks.png", dpi=120)
    print("\nWrote phase1_picks.png")


if __name__ == "__main__":
    main()
