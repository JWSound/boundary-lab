FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    BLAB_HOST=0.0.0.0 \
    BLAB_PORT=8080

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        git \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        ocl-icd-opencl-dev \
        ocl-icd-libopencl1 \
        pocl-opencl-icd \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN python -m pip install --upgrade pip \
    && python -m pip install ".[cloud,aws]"

EXPOSE 8080

ENTRYPOINT ["blab"]
CMD ["cloud-api", "--host", "0.0.0.0", "--port", "8080"]
