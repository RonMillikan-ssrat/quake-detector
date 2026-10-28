"""
picker -- owns one shard of stations: reads wf:<shard>, keeps a rolling buffer per
station, runs PhaseNet every STEP_S seconds, publishes new picks to the 'picks' stream.

Same edge and duplicate rules as phase3_live.py. Each replica must have a distinct shard
(SHARD env var or the StatefulSet ordinal in its hostname).

Metrics:
  qd_picker_infer_seconds        PhaseNet time per cycle (histogram)
  qd_picker_picks_total{phase}   new picks published
  qd_picker_stations             stations with enough buffered data
"""
import time

from obspy import Stream
from prometheus_client import Counter, Gauge, Histogram, start_http_server

import seisbench.models as sbm
from services.common import Bus, beat, cfg, my_shard

INFER = Histogram("qd_picker_infer_seconds", "PhaseNet seconds per cycle",
                  buckets=(0.1, 0.25, 0.5, 1, 2, 5, 10))
PICKS = Counter("qd_picker_picks_total", "Picks published", ["phase"])
STATIONS = Gauge("qd_picker_stations", "Stations with >=35 s buffered")


def main():
    c = cfg()
    shard = my_shard()
    start_http_server(c.metrics_port)
    bus = Bus(c.redis_url)
    bus.wait()
    model = sbm.PhaseNet.from_pretrained(c.weights)
    stream, last = f"wf:{shard}", "$"
    print(f"picker shard {shard}/{c.shards} reading {stream}", flush=True)

    buf, held = Stream(), []          # held: (station, phase, timestamp) for de-duplication
    next_t = time.time() + c.step_s
    while True:
        last, entries = bus.read(stream, last, block_ms=(next_t - time.time()) * 1000)
        for f in entries:
            buf += Bus.to_trace(f)
        if time.time() < next_t:
            continue
        next_t = max(next_t + c.step_s, time.time() + 1)
        beat()
        if not buf:
            continue

        buf.merge(method=1, fill_value=None)
        end = max(tr.stats.endtime for tr in buf)
        buf.trim(end - c.keep_s, None)
        for tr in buf:
            tr.stats.processing = []
        st = Stream([tr for tr in buf.copy().split()
                     if tr.stats.endtime - tr.stats.starttime >= 35])
        if not st:
            continue
        ends, starts = {}, {}
        for tr in st:
            sid = f"{tr.stats.network}.{tr.stats.station}"
            ends[sid] = max(ends.get(sid, tr.stats.endtime), tr.stats.endtime)
            starts[sid] = min(starts.get(sid, tr.stats.starttime), tr.stats.starttime)
        STATIONS.set(len(ends))

        tic = time.time()
        out = model.classify(st, P_threshold=c.thresh, S_threshold=c.thresh)
        INFER.observe(time.time() - tic)

        now = time.time()
        held = [h for h in held if h[2] > now - 600]
        n_new = 0
        for p in out.picks:
            sid = ".".join(p.trace_id.split(".")[:2])
            if not (starts[sid] + c.edge_s < p.peak_time < ends[sid] - c.edge_s):
                continue
            ts = p.peak_time.timestamp
            if any(h[0] == sid and h[1] == p.phase and abs(h[2] - ts) < c.dedup_s for h in held):
                continue
            held.append((sid, p.phase, ts))
            bus.publish("picks", dict(station=sid, phase=p.phase, time=repr(ts),
                                      prob=f"{p.peak_value:.3f}", found=repr(now)))
            PICKS.labels(p.phase).inc()
            n_new += 1
        print(f"shard {shard}: {len(ends)} sta, infer {time.time() - tic:.2f} s, "
              f"+{n_new} picks", flush=True)


if __name__ == "__main__":
    main()
