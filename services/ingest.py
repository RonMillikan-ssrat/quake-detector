"""
ingest -- one SeedLink connection to TexNet; every arriving miniSEED record is published
to the waveform stream of the shard that owns its station.

Singleton: run exactly one replica (two would publish every record twice).
Heartbeat is touched only when data arrives, so a stalled feed fails the liveness
probe and the orchestrator restarts the pod (which reconnects).

Metrics (port METRICS_PORT):
  qd_ingest_records_total{station}          records received
  qd_ingest_last_sample_unixtime{station}   end time of newest sample
    -> data lag per station = time() - qd_ingest_last_sample_unixtime
"""
import sys
import time

from obspy.clients.seedlink.easyseedlink import EasySeedLinkClient
from prometheus_client import Counter, Gauge, start_http_server

from services.common import Bus, beat, cfg, get_stations, publish_stations, shard_of

RECORDS = Counter("qd_ingest_records_total", "miniSEED records received", ["station"])
LAST = Gauge("qd_ingest_last_sample_unixtime", "End time of newest sample", ["station"])


class Feed(EasySeedLinkClient):
    def __init__(self, url, bus, c):
        super().__init__(url, autoconnect=False)
        self.bus, self.c, self._beat_t = bus, c, 0.0
        # ObsPy 1.5 workaround (see phase3_live.py): timeout only during connect()
        conn, orig = self.conn, self.conn.connect

        def connect_with_timeout():
            conn.timeout = 30
            try:
                return orig()
            finally:
                conn.timeout = None

        conn.connect = connect_with_timeout
        self.connect()

    def on_data(self, tr):
        sta = tr.stats.station
        self.bus.publish_trace(tr, shard_of(sta, self.c.shards), self.c.wf_maxlen)
        RECORDS.labels(sta).inc()
        LAST.labels(sta).set(tr.stats.endtime.timestamp)
        if time.time() - self._beat_t > 5:
            beat()
            self._beat_t = time.time()

    def on_seedlink_error(self):
        print("SeedLink error reported by server", flush=True)


def main():
    c = cfg()
    start_http_server(c.metrics_port)
    bus = Bus(c.redis_url)
    bus.wait()
    stations = get_stations(c)
    publish_stations(bus, stations)
    print(f"{len(stations)} stations, {c.shards} shard(s): "
          + " ".join(f"{s['sta']}:{shard_of(s['sta'], c.shards)}" for s in stations), flush=True)
    beat()
    feed = Feed(f"{c.sl_host}:{c.sl_port}", bus, c)
    for s in stations:
        feed.select_stream(s["net"], s["sta"], f"{s['band']}?")
    feed.run()                       # blocks while streaming
    print("SeedLink feed ended; exiting so the orchestrator restarts us", flush=True)
    sys.exit(1)


if __name__ == "__main__":
    main()
