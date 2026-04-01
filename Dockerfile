# syntax=docker/dockerfile:1
# ── Build stage ──────────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build
COPY requirements.txt .

RUN pip install --upgrade pip \
 && pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.12-slim

# Non-root user for Cloud Run security best-practice
RUN groupadd -r app && useradd -r -g app app

WORKDIR /app

# Copy dependencies from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY --chown=app:app . .

USER app

# Cloud Run injects PORT; Streamlit must bind to 0.0.0.0
ENV PORT=8080 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    PYTHONUNBUFFERED=1

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/_stcore/health')" || exit 1

CMD streamlit run app.py \
      --server.port=$PORT \
      --server.address=0.0.0.0 \
      --server.headless=true \
      --server.maxUploadSize=200
