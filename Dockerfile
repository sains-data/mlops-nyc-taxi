# Heroku Docker Deployment
# =========================
# Optimized for Heroku Container Registry

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    pandas \
    numpy \
    scikit-learn \
    joblib \
    pydantic

# Copy source code
COPY src/ ./src/

# Copy model and reference stats
COPY models/production_model.joblib ./models/
COPY src/serving/reference_stats.json ./src/serving/

# Expose port (Heroku assigns dynamically via $PORT)
EXPOSE 8000

# Run API with Heroku's PORT environment variable
CMD uvicorn src.serving.api:app --host 0.0.0.0 --port ${PORT:-8000}
