FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (libgomp1 is required by LightGBM C++ core)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies (pinned runtime dependencies only)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create application non-root user and state directories
RUN useradd -m -u 1000 app \
    && mkdir -p /app/state /app/data \
    && chown -R app:app /app

# Copy application files
COPY . .
RUN chown -R app:app /app

USER app

# Expose FastAPI application port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Start Uvicorn ASGI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
