# Dockerfile for the Elderly Food Vulnerability pipeline.
# Built on a slim Python image with system libs needed by geopandas/fiona/pyproj.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System libraries required for geopandas/shapely/fiona/pyproj wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgeos-dev \
        libproj-dev \
        libspatialindex-dev \
        libgdal-dev \
        gdal-bin \
        proj-bin \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# Default command runs the full pipeline.
CMD ["python", "run.py"]
