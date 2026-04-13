FROM python:3.9-slim

# System dependencies
# libgomp1: OpenMP runtime required by XGBoost at runtime
# curl: used by health check
# build-essential: required for some Python C-extension compilations
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Install Python dependencies first (maximise layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY app.py .
COPY nfl/ ./nfl/
COPY scripts/ ./scripts/
COPY assets/ ./assets/
COPY models/ ./models/

# Create EFS mount point — EFS replaces this at runtime
RUN mkdir -p /app/data && chown -R appuser:appuser /app

USER appuser

# Streamlit configuration
# ENABLE_CORS=false and ENABLE_XSRF_PROTECTION=false are required when running
# behind a reverse proxy (ALB); without these Streamlit rejects forwarded origins
ENV STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false \
    MODEL_DIR=/app/models \
    DB_PATH=/app/data/nfl_predictions.db

EXPOSE 8501

# start_period=60s: Streamlit takes 20-40s to import deps and load models.
# Without this grace period ECS kills the task during startup in a restart loop.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0", \
    "--server.headless=true"]
