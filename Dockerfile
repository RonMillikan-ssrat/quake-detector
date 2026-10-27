FROM python:3.12-slim

# CPU-only PyTorch first (avoids pulling multi-GB CUDA wheels), then the rest
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r /tmp/requirements.txt

ENV MPLBACKEND=Agg
WORKDIR /app
CMD ["python", "phase1_picks.py"]
