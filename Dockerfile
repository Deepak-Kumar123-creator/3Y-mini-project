FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    AUTOFLOW_DATA_DIR=/app/data \
    MLFLOW_ARTIFACT_DIR=/app/mlruns

WORKDIR /app

# Toolchain for source distributions without pre-built wheels (kept minimal).
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py ./
COPY templates ./templates

# Data persisted across restarts (uploads, state file, MLflow tracking DB)
# and the MLflow artifact store live outside the application code directories.
RUN mkdir -p /app/data /app/mlruns

# Run as a dedicated non-root user: the UID is fixed (10001) so bind-mounted
# volumes can be pre-created with matching ownership if required.
RUN useradd --system --create-home --home-dir /home/autoflow --uid 10001 autoflow \
    && chown -R autoflow:autoflow /app /home/autoflow
USER autoflow

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)" || exit 1

VOLUME ["/app/data", "/app/mlruns"]

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
