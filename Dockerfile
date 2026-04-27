# ────────────────────────────────────────────────────────────────────────────
# Traffic Congestion — production container
#   · Builds a reproducible environment for the offline pipeline and the
#     Streamlit dashboard.
#   · Uses Python 3.11 slim as a balance of size and wheel availability.
# ────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System deps needed for geopandas (libgdal), scipy wheels, and healthcheck curl
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Install Python deps first for better layer caching ─────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application code ──────────────────────────────────────────────────
COPY src/          ./src/
COPY app/          ./app/
COPY scripts/      ./scripts/
COPY main.py       ./
COPY README.md     ./

# Create persistent directories — these are typically volume-mounted.
RUN mkdir -p data/raw data/processed models outputs

# ── Healthcheck ────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

EXPOSE 8501

# Default: run the Streamlit dashboard. Override with e.g.:
#   docker run <image> python main.py
CMD ["streamlit", "run", "app/app.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
