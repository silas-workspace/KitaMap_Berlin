# KitaMap Berlin - Daycare Center Spatial Analysis

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GIS](https://img.shields.io/badge/GIS-GeoPandas-orange.svg)]()
[![Analysis](https://img.shields.io/badge/Analysis-Complete-brightgreen.svg)]()

> **Comprehensive spatial analysis of daycare center coverage in Berlin using GIS technology, including demographic forecasting until 2034**

**🌐 [View Interactive Dashboard](https://pinea.app.carto.com/map/81885962-c7a8-4639-8124-372e0caa6e60)**

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/silas-workspace/KitaMap_Berlin.git
cd KitaMap_Berlin

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the complete analysis
python main.py

# 4. Explore Jupyter notebooks
jupyter notebook notebooks/
```

## Project Overview

KitaMap Berlin is a data-driven project that analyzes the spatial distribution and accessibility of daycare centers in Berlin. The project combines GIS workflows, demographic forecasting, and interactive visualization to support urban planning and policy decisions.

### Key Features

- **📊 Coverage Analysis**: District-level assessment of current daycare availability
- **🔮 Demographic Forecasting**: Population-based demand prediction until 2034
- **📍 Gap Identification**: Detection of underserved areas and hotspots
- **🚶‍♀️ Accessibility Analysis**: Walking-distance catchment area calculations
- **🌱 Environmental Integration**: Proximity analysis to green spaces and water areas
- **📱 Interactive Visualization**: Web-based dashboard with CARTO integration

## Technology Stack

### Core Technologies
- **Python 3.8+** - Primary programming language
- **GeoPandas** - Spatial data analysis and manipulation
- **Pandas & NumPy** - Data processing and numerical computations
- **Shapely** - Geometric operations and spatial calculations

### GIS & Mapping
- **CARTO** - Interactive web-based visualization platform
- **OpenStreetMap** - Base map data and point-of-interest extraction
- **OpenRouteService API** - Isochrone and routing calculations
- **OSMium** - High-performance OSM data processing

### Analysis & Forecasting
- **Prophet** - Time series forecasting for demographic predictions
- **Statsmodels** - Statistical modeling and regression analysis
- **Scikit-learn** - Machine learning utilities
- **Matplotlib & Seaborn** - Statistical visualizations

### Development Environment
- **Jupyter Notebooks** - Interactive analysis and documentation
- **TQDM** - Progress bars for long-running operations
- **Python-dotenv** - Environment variable management
- **Ruff / Pyright / Nox** - Linting, type checking, and task automation

## Project Structure

```text
KitaMap_Berlin/
├── AGENTS.md
├── LICENSE
├── README.md
├── main.py
├── noxfile.py
├── pyproject.toml
├── requirements.txt
├── run_analysis.py
├── data/
│   ├── external/
│   │   └── README.md
│   ├── processed/
│   │   └── README.md
│   ├── raw/
│   │   └── README.md
│   └── results/
│       └── README.md
├── docs/
│   └── methodology.md
├── notebooks/
│   ├── 01_daycare_data_processing.ipynb
│   ├── 01_daycare_data_processing.py
│   ├── 02_demographic_forecasting.ipynb
│   ├── 02_demographic_forecasting.py
│   ├── 03_district_analysis.ipynb
│   ├── 03_district_analysis.py
│   ├── 04_spatial_analysis.ipynb
│   ├── 04_spatial_analysis.py
│   └── README.md
└── src/
    ├── __init__.py
    ├── config.py
    └── spatial_analysis.py
```

## Usage Examples

### Command Line Analysis
```bash
# Run complete spatial analysis
python main.py

# Extract only OSM areas (green spaces, water bodies)
python run_analysis.py --osm-only

# Generate isochrones only (requires API key)
python run_analysis.py --isochrones-only --api-key YOUR_ORS_KEY
```

### Jupyter Notebooks
The project includes four analysis notebooks:

1. **`01_daycare_data_processing.ipynb`** - Daycare data cleaning and capacity estimation
2. **`02_demographic_forecasting.ipynb`** - Population forecasting until 2034
3. **`04_spatial_analysis.ipynb`** - OSM area extraction and isochrone generation
4. **`03_district_analysis.ipynb`** - District coverage analysis and CARTO exports

Run them in this order: **01 → 02 → 04 → 03**.

### API Configuration
For isochrone generation, set your OpenRouteService API key:
```bash
export OPENROUTESERVICE_API_KEY="your_api_key_here"
```

## Data Sources

| Source | Description | Usage |
|--------|-------------|-------|
| **OpenStreetMap** | Daycare locations and metadata | Geocoding, facility data extraction |
| **Berlin Geoportal** | Administrative boundaries | District boundaries, spatial references |
| **Berlin Statistics Office** | Demographic data | Population forecasts and trends |
| **OpenRouteService** | Routing and accessibility | Isochrone calculations, walking distances |

## Methodology

### 1. Data Collection & Processing
- Automated extraction of daycare locations from OpenStreetMap
- Capacity estimation using area-based regression and district-specific medians
- Integration of demographic data at district level

### 2. Spatial Analysis
- **Catchment Areas**: 500 m walking-distance isochrones around facilities
- **Accessibility Analysis**: Route-based calculations via OpenRouteService API
- **Coverage Assessment**: District-level availability metrics

### 3. Forecasting
- **Time Series Modeling**: Prophet-based population predictions until 2034
- **Demand Estimation**: Future daycare needs based on demographic trends
- **Gap Analysis**: Identification of supply-demand mismatches

### 4. Visualization
- **Interactive Dashboard**: CARTO-powered web visualization
- **Statistical Graphics**: Analysis reports with Matplotlib/Seaborn
- **Jupyter Integration**: Reproducible analysis workflows

For full methodological detail, see [docs/methodology.md](docs/methodology.md).

## Key Results

### Current Coverage (2024)
- **Over-supplied Districts**: Charlottenburg-Wilmersdorf, Steglitz-Zehlendorf
- **Under-supplied Districts**: Neukölln, Marzahn-Hellersdorf
- **Critical Areas**: Identified hotspots with significant coverage gaps

### 2034 Forecast
- **Population Growth**: Rising demand in several districts
- **Additional Demand**: Need for additional daycare spots by 2034
- **Priority Areas**: Districts requiring targeted intervention

## Applications

- **Urban Planning**: Evidence-based location planning for new facilities
- **Policy Making**: Data-driven resource allocation decisions
- **Research**: Methodological framework for similar urban analyses
- **Public Transparency**: Clear visualization of service coverage

## Installation

### Prerequisites
- Python 3.8 or higher
- Git

### Setup
```bash
# 1. Clone the repository
git clone https://github.com/silas-workspace/KitaMap_Berlin.git
cd KitaMap_Berlin

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run analysis
python main.py
```

Before running the full workflow:
- Download `berlin-latest.osm.pbf` from https://download.geofabrik.de/europe/germany/berlin.html and place it in `data/raw/`. The file is not tracked because it is large.
- Ensure `data/raw/daycare_centers_osm.geojson` is present before running Notebook 01.

## License

MIT. See [LICENSE](LICENSE).

## Author

**Silas Pignotti**
- GitHub: [@silas-workspace](https://github.com/silas-workspace)
- Project: [KitaMap Berlin](https://github.com/silas-workspace/KitaMap_Berlin)

## Acknowledgments

- **OpenStreetMap Community** for comprehensive geospatial data
- **Berlin Geoportal** for administrative boundaries
- **Berlin Statistics Office** for demographic data
- **CARTO** for visualization platform
- **OpenRouteService** for routing and accessibility APIs
