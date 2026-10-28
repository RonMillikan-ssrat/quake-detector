"""
Shared pieces for the quake-detector services: configuration, the Redis Streams bus,
station sharding, the station list, and a liveness heartbeat.

Every setting comes from an environment variable (default in parentheses), so the same
image runs under docker compose and Kubernetes; in Kubernetes they come from a ConfigMap.
"""
import json
import os
import re
import socket
import time
import zlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import redis
from obspy import Trace, UTCDateTime

HEARTBEAT = Path("/tmp/heartbeat")


def _env(name, default, cast=str):
    v = os.environ.get(name)
    return cast(v) if v not in (None, "") else default


def cfg():
    return SimpleNamespace(
        box=dict(minlatitude=_env("BOX_MINLAT", 31.0, float),
                 maxlatitude=_env("BOX_MAXLAT", 32.2, float),
                 minlongitude=_env("BOX_MINLON", -104.6, float),
                 maxlongitude=_env("BOX_MAXLON", -103.0, float)),
        sl_host=_env("SL_HOST", "rtserve.beg.utexas.edu"),
        sl_port=_env("SL_PORT", 18000, int),
        fdsn_url=_env("FDSN_URL", "http://rtserve.beg.utexas.edu"),
        redis_url=_env("REDIS_URL", "redis://localhost:6379/0"),
        pg_dsn=_env("PG_DSN", "postgresql://quake:quake@localhost:5432/quake"),
        shards=_env("SHARDS", 1, int),
        weights=_env("WEIGHTS", "instance"),
        thresh=_env("THRESH", 0.3, float),
        keep_s=_env("KEEP_S", 90.0, float),
        step_s=_env("STEP_S", 10.0, float),
        edge_s=_env("EDGE_S", 5.0, float),
        dedup_s=_env("DEDUP_S", 0.5, float),
        assoc_s=_env("ASSOC_S", 300.0, float),
        settle_s=_env("SETTLE_S", 30.0, float),
        vp=_env("VP", 5.5, float),
        vs=_env("VS", 3.1, float),
        tol=_env("TOL", 1.0, float),
        n_picks=_env("N_PICKS", 6, int),
        n_ps=_env("N_PS", 3, int),
        metrics_port=_env("METRICS_PORT", 9100, int),
        wf_maxlen=_env("WF_MAXLEN", 20000, int),
    )


# ---------------------------------------------------------------- sharding
def shard_of(sta, n):
    """Stable station -> shard. (Python's hash() is randomized per process, so a
    checksum is used: ingest and every picker must agree across restarts.)"""
    return zlib.crc32(sta.encode()) % n


def my_shard():
    """SHARD env var, else the ordinal at the end of the hostname: a StatefulSet
    names its pods picker-0, picker-1, ... so each knows its shard without config."""
    if os.environ.get("SHARD"):
        return int(os.environ["SHARD"])
    m = re.search(r"-(\d+)$", socket.gethostname())
    return int(m.group(1)) if m else 0


# ---------------------------------------------------------------- bus
class Bus:
    """Redis Streams. Waveforms go to wf:<shard>, picks to 'picks', events to 'events'.
    Streams are capped (approximate MAXLEN) so memory stays bounded."""

    MAX_BLOCK_MS = 5000

    def __init__(self, url):
        # bytes in/out: waveform payloads are binary.
        # redis-py 8 defaults socket_timeout to 5 s; a blocking XREAD that waits longer
        # than that raises TimeoutError. Keep the socket timeout well above MAX_BLOCK_MS.
        self.r = redis.Redis.from_url(url, socket_timeout=30, socket_connect_timeout=10)

    def wait(self, timeout=120):
        t_end = time.time() + timeout
        while True:
            try:
                self.r.ping()
                return
            except redis.exceptions.ConnectionError:
                if time.time() > t_end:
                    raise
                print("waiting for redis...", flush=True)
                time.sleep(2)

    def publish_trace(self, tr, shard, maxlen):
        # float32: PhaseNet normalizes each window, so the low bits lost for counts
        # above 2^24 do not matter, and it halves the payload vs float64
        self.r.xadd(f"wf:{shard}", {
            "id": tr.id,
            "t0": repr(tr.stats.starttime.timestamp),
            "sr": repr(tr.stats.sampling_rate),
            "data": np.asarray(tr.data, dtype=np.float32).tobytes(),
        }, maxlen=maxlen, approximate=True)

    @staticmethod
    def to_trace(f):
        net, sta, loc, cha = f[b"id"].decode().split(".")
        return Trace(np.frombuffer(f[b"data"], dtype=np.float32).copy(), header=dict(
            network=net, station=sta, location=loc, channel=cha,
            starttime=UTCDateTime(float(f[b"t0"])), sampling_rate=float(f[b"sr"])))

    def publish(self, stream, d, maxlen=100000):
        self.r.xadd(stream, {k: str(v) for k, v in d.items()}, maxlen=maxlen, approximate=True)

    def read(self, stream, last_id, block_ms, count=1000):
        """Entries after last_id ('$' = only new ones). Returns (new_last_id, [fields])."""
        block = min(max(1, int(block_ms)), self.MAX_BLOCK_MS)
        res = self.r.xread({stream: last_id}, count=count, block=block)
        out = []
        for _, entries in res or []:
            for eid, f in entries:
                last_id = eid
                out.append(f)
        return last_id, out


# ---------------------------------------------------------------- stations
BANDS = ("HH", "EH", "HN")


def get_stations(c):
    """TX stations in the box with a usable band, filtered to what the SeedLink server
    streams (falls back to all if that large INFO reply times out)."""
    from obspy.clients.fdsn import Client as FDSNClient
    from obspy.clients.seedlink.basic_client import Client as SLInfoClient
    now = UTCDateTime()
    inv = FDSNClient(base_url=c.fdsn_url).get_stations(
        network="TX", channel="HH?,EH?,HN?", starttime=now - 86400, endtime=now,
        level="channel", **c.box)
    streamed = None
    for attempt in range(3):
        try:
            streamed = {(n, s) for n, s in SLInfoClient(c.sl_host, c.sl_port, timeout=60)
                        .get_info(network="TX", level="station")}
            break
        except Exception as e:
            print(f"SeedLink station list attempt {attempt + 1} failed: {e}", flush=True)
            time.sleep(5)
    stas = []
    for net in inv:
        for sta in net:
            band = next((b for b in BANDS if b in {ch.code[:2] for ch in sta}), None)
            if band and (streamed is None or (net.code, sta.code) in streamed):
                stas.append(dict(net=net.code, sta=sta.code, band=band, lat=sta.latitude,
                                 lon=sta.longitude, elev_m=sta.elevation))
    return sorted(stas, key=lambda s: s["sta"])


def publish_stations(bus, stations):
    bus.r.set("stations", json.dumps(stations))


def load_stations(bus, timeout=600):
    """Wait for ingest to publish the station list (single source of truth)."""
    t_end = time.time() + timeout
    while True:
        raw = bus.r.get("stations")
        if raw:
            return json.loads(raw)
        if time.time() > t_end:
            raise RuntimeError("no station list in redis")
        print("waiting for station list from ingest...", flush=True)
        time.sleep(5)


def beat():
    """Liveness heartbeat: a probe fails if this file goes stale."""
    HEARTBEAT.touch()
