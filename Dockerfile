# Wildfire Nexus Core - Production Docker Image
# Base image with Python 3.11 and system dependencies for GDAL/Rasterio
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # GDAL configuration
    CPLUS_INCLUDE_PATH=/usr/include/gdal \
    C_INCLUDE_PATH=/usr/include/gdal \
    # App settings
    APP_HOME=/app \
    DATA_DIR=/app/data \
    CACHE_DIR=/app/data/cache \
    OUTPUT_DIR=/app/data/outputs

# Install system dependencies required for geospatial libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    # GDAL core libraries
    gdal-bin \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    libffi-dev \
    # Build tools
    build-essential \
    cmake \
    # Utilities
    curl \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Get GDAL version to match python bindings
RUN export CPLUS_INCLUDE_PATH=$(gdal-config --cflags) \
    && export C_INCLUDE_PATH=$(gdal-config --cflags)

# Set working directory
WORKDIR ${APP_HOME}

# Copy application code before editable install
COPY pyproject.toml README.md ./
COPY app/ ./app/
COPY config/ ./config/
COPY scripts/ ./scripts/
COPY data/fixtures/ ./data/fixtures/
COPY tests/ ./tests/
COPY .env.example ./.env

# Install Python dependencies
# Note: rasterio, geopandas will compile against system GDAL
RUN pip install --upgrade pip && \
    pip install -e ".[dev]"

# Create necessary directories for runtime data
RUN mkdir -p ${CACHE_DIR} ${OUTPUT_DIR} && \
    chmod -R 755 ${DATA_DIR}

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
