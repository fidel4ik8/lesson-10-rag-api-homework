# Single-stage build. Multi-stage saved nothing here because the deps layer
# was the one we were copying around. Single-stage + aggressive cleanup is
# both smaller and easier to reason about.

FROM python:3.11-slim

# HF_HUB_OFFLINE is set AFTER the model download step below — otherwise
# the build step that pre-downloads the model can't reach huggingface.co.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf-cache \
    SENTENCE_TRANSFORMERS_HOME=/opt/hf-cache

WORKDIR /app

# Tiny set of system libs the binary wheels link against.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libstdc++6 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only torch first. sentence-transformers depends on it and would
# otherwise pull the GPU wheel (~2 GB of CUDA libs).
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1+cpu

COPY pyproject.toml ./
COPY app ./app
COPY scripts ./scripts

# Project + remaining deps. Torch above already satisfies the dep, so this
# step does not reach for the CPU index again.
RUN pip install .

# Pre-download the embedding model so cold starts don't depend on Hugging
# Face being reachable from the runtime.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Cache is warm now — make the runtime offline so any code path that touches
# HF doesn't try to phone home.
ENV HF_HUB_OFFLINE=1

# Strip bytecode caches only. Do NOT delete `tests/`, `test/`, or `testing/`
# directories — several packages (numpy, torch) ship runtime code under those
# names and break at import if it's missing. The saving is not worth the
# debugging.
RUN set -eux; \
    find /usr/local/lib/python3.11 -depth \
        \( -name "__pycache__" -type d \
        -o -name "*.pyc" \
        -o -name "*.pyo" \) \
        -exec rm -rf {} +; \
    rm -rf /usr/local/lib/python3.11/site-packages/torch/include \
           /usr/local/lib/python3.11/site-packages/torch/lib/*.lib 2>/dev/null || true

COPY data ./data

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
