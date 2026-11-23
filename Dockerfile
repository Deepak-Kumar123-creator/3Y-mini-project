FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    AUTOFLOW_DATA_DIR=/app/data \
    MLFLOW_ARTIFACT_DIR=/app/mlruns

WORKDIR /app

# Install Python dependencies
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt

# Copy required application files
COPY main.py visualization.py ./
COPY meta_learning/ ./meta_learning/
COPY templates/ ./templates/

# Create persistent storage directories and configure permissions
RUN mkdir -p /app/data/meta_data /app/mlruns \
    && useradd --system --create-home \
       --home-dir /home/autoflow --uid 10001 autoflow \
    && chown -R autoflow:autoflow /app /home/autoflow

# Run application as a non-root user
USER autoflow

EXPOSE 8000

# Application health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)" || exit 1

# Persistent storage
VOLUME ["/app/data", "/app/mlruns"]

# Start FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
