# quake-detector

Real-time ML earthquake detection on live seismic data, with induced-seismicity
monitoring for the Permian Basin. Built with SeisBench (PhaseNet), ObsPy, and
public TexNet / USGS FDSN services; containerized for Kubernetes.

## Status
- Phase 1: pretrained PhaseNet picks on an M4.0 Delaware Basin event (2026-10-11);
  also detected the cataloged M2.9 aftershock 69 s later.

## Run
    docker build -t quake-detector .
    docker run --rm -v "$PWD":/app -v seisbench-cache:/root/.seisbench quake-detector
