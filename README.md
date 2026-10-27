# quake-detector

Real-time ML earthquake detection on live seismic data, with induced-seismicity
monitoring for the Permian Basin. Built with SeisBench (PhaseNet), ObsPy, and
public TexNet / USGS FDSN services; containerized for Kubernetes.

## Status
- Phase 1: pretrained PhaseNet picks on an M4.0 Delaware Basin event (2026-10-11);
  also detected the cataloged M2.9 aftershock 69 s later.

## Phase 2: one hour of continuous data (2026-10-11 07:30–08:30 UTC)

33 TexNet stations, pretrained PhaseNet (`instance` weights), PyOcto association,
homogeneous velocity model (Vp 5.5, Vs 3.1 km/s).

| Result | Value |
|---|---|
| Events detected | 25 |
| Catalog events in window (TexNet via USGS ComCat, no magnitude floor) | 6 (M1.7–M4.0) |
| Catalog recall | 6/6 |
| Uncataloged detections | 19 |
| …real by visual review AND waveform similarity above a cross-station null | 12 |
| …supported by one method only | 7 |
| …rejected by both | 0 |
| Inference cost | 0.34 s CPU per station-hour |

Known limitations
- Locations of well-recorded events are offset ~5 km from catalog, consistently in one
  direction; attributed to the homogeneous velocity model. Depths are grid-quantized
  and not meaningful.
- Visual review was a single pass in time order, blind to correlation results but not
  to catalog status; reviewer calibration drift is possible.

## Run
    docker build -t quake-detector .
    docker run --rm -v "$PWD":/app -v seisbench-cache:/root/.seisbench quake-detector
