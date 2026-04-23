FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files flat (no subdirectories)
COPY __init__.py .
COPY models.py .
COPY environment.py .
COPY server.py .
COPY inference.py .
COPY openenv.yaml .
COPY emails.py .
COPY grader.py .
COPY test_environment.py .
COPY train_grpo.py .
COPY train_grpo_colab.py .
COPY pyproject.toml .
COPY static/ ./static/

EXPOSE 7860

HEALTHCHECK --interval=10s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]