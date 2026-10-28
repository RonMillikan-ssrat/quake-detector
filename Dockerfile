FROM python:3.12-slim

# CPU-only PyTorch first (avoids pulling multi-GB CUDA wheels), then the rest
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r /tmp/requirements.txt

ENV MPLBACKEND=Agg \
    PYTHONUNBUFFERED=1 \
    SEISBENCH_CACHE_ROOT=/opt/seisbench

# Bake the model weights into the image: pods start without downloading,
# and any user id can read them.
RUN python -c "import seisbench.models as sbm; sbm.PhaseNet.from_pretrained('instance')" \
 && chmod -R a+rX /opt/seisbench

WORKDIR /app
# Code is copied in for compose/Kubernetes; the qd alias still mounts $PWD over it.
COPY . /app
CMD ["python", "phase1_picks.py"]
