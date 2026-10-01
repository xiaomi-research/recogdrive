# AlpaSim end-to-end challenge driver: recogdrive.closedloop.alpasim serving one trained checkpoint, offline.
# Build from the repo root with an AlpaSim checkout (for its protos) and the weights in the build context:
#   git clone https://github.com/NVlabs/alpasim docker/alpasim/alpasim
#   docker/alpasim/weights/model.ckpt (and model_vlm/ if the VLM was trained), docker/alpasim/weights/vlm/ (base VLM)
#   docker build -f docker/alpasim/Dockerfile -t recogdrive-alpasim .
# Agent overrides go into the CMD, e.g. agent.cameras for a multi-view model. Challenge limits: image <= 40 GiB,
# <= 16 GiB GPU memory per instance, no network, read-only root with a writable /tmp.
FROM python:3.9-slim

RUN apt-get update && apt-get install -y --no-install-recommends build-essential git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt requirements.txt
COPY navsim1.1/requirements.txt navsim1.1/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt grpcio==1.80.0 grpcio-tools==1.62.3

COPY docker/alpasim/alpasim/src/grpc /tmp/alpasim/src/grpc
COPY scripts/infer/build_alpasim_grpc.sh /tmp/build_alpasim_grpc.sh
RUN bash /tmp/build_alpasim_grpc.sh /tmp/alpasim /opt/alpasim_grpc && rm -rf /tmp/alpasim /tmp/build_alpasim_grpc.sh

COPY recogdrive recogdrive
COPY navsim1.1 navsim1.1
COPY scripts scripts
COPY docker/alpasim/weights weights

RUN useradd --system --home-dir /app --shell /usr/sbin/nologin alpasim
ENV PYTHONPATH=/opt/alpasim_grpc HOME=/tmp HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    TORCHINDUCTOR_CACHE_DIR=/tmp/inductor TRITON_CACHE_DIR=/tmp/triton NAVSIM_EXP_ROOT=/tmp/exp \
    ALPASIM_DRIVER_HOST=0.0.0.0 ALPASIM_DRIVER_PORT=6789 \
    SERVE=alpasim CHECKPOINT=/app/weights/model.ckpt VLM_PATH=/app/weights/vlm
EXPOSE 6789
USER alpasim
CMD ["bash", "scripts/infer/run_recogdrive_closedloop.sh"]
