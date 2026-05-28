FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

WORKDIR /workspace

# system dependencies
RUN apt-get update && apt-get install -y \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    libnetcdf-dev \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

# SSL — portable across machines
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt

# W&B — set at runtime via -e WANDB_API_KEY=your_key
ENV WANDB_MODE=offline

# match host user to avoid permission issues with mounted volumes
ARG UID=1000
ARG GID=1000
RUN groupadd -g $GID appgroup && useradd -u $UID -g $GID -m appuser

# Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# source code only — data and outputs mounted at runtime
COPY machine-learning-module/src/     ./src/
COPY machine-learning-module/config/  ./config/

# runtime
# docker run \
#   -e WANDB_API_KEY=your_key \
#   -v /path/to/data:/workspace/data \
#   -v /path/to/outputs:/workspace/outputs \
#   your_image python src/main.py --config config/exp_001.yaml