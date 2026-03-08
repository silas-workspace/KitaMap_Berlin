# Jupyter Notebooks - KitaMap Berlin

This directory contains Jupyter notebooks for exploratory analysis and reproducible project workflows.

## Notebook Overview

### 01_daycare_data_processing.ipynb
**Main Goal**: Process daycare data from OpenStreetMap and estimate missing capacities.

**Core Tasks**:
- Integrate OSM daycare data with Berlin district boundaries
- Estimate missing capacities for polygon and point geometries
- Validate and scale results to the project target capacity
- Export the processed daycare dataset

**Output**: `../data/processed/daycare_centers_processed.geojson`

### 02_demographic_forecasting.ipynb
**Main Goal**: Forecast population-based daycare demand until 2034.

**Core Tasks**:
- Analyze population development by district
- Compare Prophet and Exponential Smoothing forecasts
- Export district-level forecasts for later analysis

**Output**: `../data/processed/population_forecast_2024_2034.csv`

### 03_district_analysis.ipynb
**Main Goal**: Analyze district coverage and prepare CARTO export layers.

**Core Tasks**:
- Prepare district geometries for analysis
- Aggregate daycare supply metrics by district
- Calculate coverage, category, and trend outputs
- Export GeoJSON layers for visualization

**Output**: Multiple files in `../data/external/`

### 04_spatial_analysis.ipynb
**Main Goal**: OSM area extraction and isochrone generation.

**Core Tasks**:
- Extract green spaces and water bodies from `berlin-latest.osm.pbf`
- Generate 500 m walking-distance isochrones via OpenRouteService API
- Remove overlapping isochrone areas for coverage calculations

**Prerequisites**: `OPENROUTESERVICE_API_KEY` environment variable and `data/raw/berlin-latest.osm.pbf`
**Output**: Files in `../data/results/`

## Usage

Run notebooks in this order: **01 → 02 → 04 → 03**.

Notebook 04 must complete before Notebook 03 can run the isochrone-based analysis.

### Prerequisites
```bash
pip install -r ../requirements.txt
jupyter notebook
```

### Data Structure
- Raw data: `../data/raw/`
- Processed data: `../data/processed/`
- Analysis results: `../data/results/`
- External exports: `../data/external/`
