"""Shared configuration constants for KitaMap Berlin."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Coordinate reference systems
PROJ_CRS = "EPSG:32633"
EXPORT_CRS = "EPSG:4326"

# Berlin district boundaries (WFS)
WFS_URL = (
    "https://gdi.berlin.de/services/wfs/alkis_bezirke"
    "?REQUEST=GetCapabilities&SERVICE=wfs"
)

# Capacity calibration
TARGET_CAPACITY = 200_000

# OpenRouteService parameters
ISOCHRONE_RANGE_M = 500
MAX_API_REQUESTS = 450

# Data directories
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"

# Input files
DAYCARE_OSM_FILE = RAW_DIR / "daycare_centers_osm.geojson"
OSM_PBF_FILE = RAW_DIR / "berlin-latest.osm.pbf"
POPULATION_FILE = RAW_DIR / "population_development_2015_2024.csv"

# Processed files
DAYCARE_PROCESSED_FILE = PROCESSED_DIR / "daycare_centers_processed.geojson"
POPULATION_FORECAST_FILE = PROCESSED_DIR / "population_forecast_2024_2034.csv"

# Results files
ISOCHRONES_FILE = RESULTS_DIR / "isochrones.geojson"
ISOCHRONES_CLEAN_FILE = RESULTS_DIR / "isochrones_overlapping.geojson"
GREEN_AREAS_FILE = RESULTS_DIR / "berlin_green_areas.geojson"
WATER_AREAS_FILE = RESULTS_DIR / "berlin_water_areas.geojson"
