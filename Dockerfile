# ── Stage 1: Build ────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Stage 2: App ──────────────────────────────────────────────────────────────
FROM base AS app

WORKDIR /app

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY models.py .
COPY environment.py .
COPY server.py .
COPY inference.py .
COPY openenv.yaml .

# Copy data and graders
COPY data/ ./data/
COPY graders/ ./graders/

# Create __init__.py files for modules
RUN touch data/__init__.py graders/__init__.py

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=10s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Default command: start the OpenEnv server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]

# ── Hugging Face Spaces compatibility ─────────────────────────────────────────
# HF Spaces uses port 7860 by default. Override with: -e PORT=7860
# The server reads PORT from environment:
#   uvicorn server:app --host 0.0.0.0 --port ${PORT:-8080}
