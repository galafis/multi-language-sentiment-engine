# Multi-Language Sentiment Analysis Engine - Production Dockerfile
# Base image with Python 3.9 slim for better performance
FROM python:3.9-slim-bullseye

# Metadata labels for better container management
LABEL maintainer="Gabriel Demetrios Lafis <gabriel@galafis.dev>" \
      description="Multi-language sentiment analysis engine with transformer models" \
      version="1.0.0" \
      org.opencontainers.image.source="https://github.com/galafis/multi-language-sentiment-engine"

# Set environment variables for Python optimization
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create non-root user for security
RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

# Install system dependencies and clean up in single layer
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        libgomp1 \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies with optimizations
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir gunicorn[gevent] uvloop

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY scripts/ ./scripts/

# Create necessary directories and set permissions
RUN mkdir -p /app/logs /app/models /app/cache \
    && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port for the API service
EXPOSE 8000

# Health check configuration
HEALTHCHEK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command - can be overridden for different services
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "--worker-connections", "1000", "--max-requests", "1000", "--max-requests-jitter", "50", "--preload", "--access-logfile", "-", "--error-logfile", "-", "src.api.rest_api:app"]

# Multi-stage build for smaller production image (optional enhancement)
# Uncomment the following lines for multi-stage build
# FROM python:3.9-slim-bullseye as production
# COPY --from=0 /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
# COPY --from=0 /usr/local/bin /usr/local/bin
# COPY --from=0 /app /app
# USER appuser
# WORKDIR /app
# CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "src.api.rest_api:app"]
