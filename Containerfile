# Multi-stage build. The builder installs dependencies into a virtualenv; the runtime
# copies only that venv and the application, so compilers and build headers never
# reach the shipped image.
#
# Build:  podman build --format docker -t eta-api:latest -f Containerfile .
# Run:    podman-compose up          (preferred - it applies the healthcheck)
#
# --format docker is required for the HEALTHCHECK below to be recorded at all: Podman
# defaults to the OCI format, which has no healthcheck field and discards it silently.
# Note that `podman run` still does not *inherit* the recorded healthcheck; compose
# declares it explicitly, and a plain run needs --health-cmd to activate it.

FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# LightGBM links against libgomp at build time.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Dependencies are copied and installed before the source so that editing code does
# not invalidate the (slow) dependency layer. Only the serving set is installed:
# DVC, pytest, ruff, Streamlit, Plotly and matplotlib are unreachable from the request
# path and would be pure weight and attack surface in a shipped image.
COPY requirements-serving.txt .
RUN pip install --upgrade pip && pip install -r requirements-serving.txt


FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

# libgomp is a runtime requirement of LightGBM, not just a build one.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Serving as a non-root user: a container compromise should not yield root inside it.
RUN useradd --create-home --uid 10001 appuser

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY src/ ./src/
COPY params.yaml ./

# Model and feature artefacts. models/trained/model is the standalone export of the
# selected model; the mlruns tracking store is deliberately NOT copied, since it holds
# every candidate from every run and is several hundred megabytes.
COPY models/feature_pipeline/ ./models/feature_pipeline/
COPY models/trained/ ./models/trained/

RUN mkdir -p /app/monitoring/data && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Readiness, not liveness: the container is only useful once the model has loaded.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/ready || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
