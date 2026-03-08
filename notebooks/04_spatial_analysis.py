# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 04 — Spatial Analysis
# > OSM area extraction and isochrone generation using OpenRouteService API.
#
# ---
#
# Run this notebook after completing the data preparation notebooks. Requires:
# - `data/raw/berlin-latest.osm.pbf` (download from Geofabrik; see `data/raw/README.md`)
# - `OPENROUTESERVICE_API_KEY` in the environment or a local `.env` file

# %% [markdown]
# ## Setup

# %%
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path("../src").resolve()))

from config import OSM_PBF_FILE, RESULTS_DIR
from spatial_analysis import extract_osm_areas, generate_isochrones

load_dotenv()

# %% [markdown]
# ## 1. OSM Area Extraction
#
# Read the Berlin OSM PBF and extract green areas and water bodies.
# Outputs are written to `data/results/berlin_green_areas.geojson` and `data/results/berlin_water_areas.geojson`.

# %%
green_path, water_path = extract_osm_areas(OSM_PBF_FILE, RESULTS_DIR)

# %% [markdown]
# ## 2. Isochrone Generation
#
# Compute 500 m walking-distance isochrones for all daycare centers via OpenRouteService.
# Outputs are written to `data/results/isochrones.geojson` and `data/results/isochrones_overlapping.geojson`.

# %%
isochrones_path = generate_isochrones()

# %% [markdown]
# ## Results / Summary
#
# These outputs are consumed by the district analysis notebook and by the dashboard export workflow.

# %%
print(f"Green areas: {green_path}")
print(f"Isochrones:  {isochrones_path}")
