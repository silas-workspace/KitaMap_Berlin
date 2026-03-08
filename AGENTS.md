# KitaMap Berlin — Agent Context

## Project

Academic project (BHT Berlin, 2024) analyzing daycare center (Kita) supply in Berlin.
GIS-based pipeline: OSM data extraction → capacity estimation (regression) → demographic
forecasting (Prophet, ETS, Ensemble) → accessibility analysis (OpenRouteService isochrones)
→ CARTO dashboard visualization.

An accompanying academic report (PDF) exists. **The numerical results of all analyses must
be preserved exactly** — this is the #1 constraint for all edits.

## Repository Structure

```
KitaMap_Berlin/
├── notebooks/              # Analysis notebooks (run in order: 01 → 02 → 03)
│   ├── 01_daycare_data_processing.ipynb   # OSM ETL + capacity estimation
│   ├── 02_demographic_forecasting.ipynb   # Prophet/ETS/Ensemble forecasting
│   └── 03_district_analysis.ipynb         # Coverage analysis + CARTO exports
├── src/
│   └── spatial_analysis.py   # Isochrone generation + OSM area extraction (ORS API)
├── data/
│   ├── raw/        # Input data (OSM PBF, population CSV) — do not modify contents
│   ├── processed/  # Intermediate outputs (daycare_centers_processed.geojson, forecast CSV)
│   ├── results/    # Analysis outputs (isochrones, green/water areas)
│   └── external/   # CARTO-ready exports (GeoJSON for dashboard)
├── docs/
│   └── methodology.md   # Methodological documentation — keep, it is substantive
├── main.py          # Entry point — delegates to run_analysis.py
├── run_analysis.py  # CLI with --osm-only / --isochrones-only flags
├── pyproject.toml   # Project metadata + tooling config
└── requirements.txt # Runtime dependencies
```

## Key Conventions

- **Language**: English for all code, comments, docstrings, and markdown. German strings
  that are data values (column names, category labels output to GeoJSON) must NOT be
  changed — they feed the CARTO dashboard and would break visualizations.
- **Notebooks**: Edit only via Jupytext `.py` pairs (percent format). Never edit `.ipynb`
  directly. Sync with `jupytext --sync <notebook>.ipynb` after every edit. Outputs in
  `.ipynb` must not be cleared — they are the "known-good" reference.
- **CRS**: EPSG:32633 (UTM Zone 33N) for internal computations; EPSG:4326 for exports.
- **Data flow**: NB01 → NB02 → NB03. Each notebook reads from prior outputs. Do not
  reorder or combine.
- **API key**: OpenRouteService key via `OPENROUTESERVICE_API_KEY` env var. Never hardcode.

## Constraints

- **No logic changes**: no algorithm changes, no control flow changes, no new computations
- **No restructuring**: directory structure is final
- **No runtime dependency changes**: requirements.txt content is locked
- **Results identical**: notebooks must produce the same outputs when re-run
- **German data values are untouchable**: category strings ('Optimal versorgt',
  'Unterversorgt', etc.), column names ('Bezirk', 'Gesamtkapazität', etc.), and any
  value that ends up in exported GeoJSON files must remain as-is

For global coding standards, see ~/.pi/agent/AGENTS.md
