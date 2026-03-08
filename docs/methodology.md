# Methodology - KitaMap Berlin

## Overview

KitaMap Berlin employs a multi-stage analytical methodology for comprehensive assessment of daycare center coverage in Berlin. The approach combines geospatial analysis, statistical modeling, and time series forecasting.

## 1. Data Sources & Processing

### Primary Data Sources
- **OpenStreetMap (OSM)**: Daycare locations and facility metadata
- **Berlin Geoportal**: Administrative boundaries (districts)
- **Berlin Statistics Office**: Demographic and population data

### Data Processing Pipeline
1. **Geocoding & Validation**: Spatial data quality assurance
2. **Administrative Assignment**: Spatial joining to district boundaries
3. **Format Harmonization**: Standardization across different data sources

## 2. Capacity Estimation

### Area-Based Regression (Building Footprints)
For daycare centers with mapped building footprints:

```
Capacity = α + β × Building_Area + ε
```

**Model Parameters**:
- α (Base capacity): ~69 spots
- β (Area effect): ~0.0132 spots/m²
- Statistical significance: p < 0.001

### District-Specific Median Estimation (Point Features)
For point-based daycare locations:

```
Estimated_Capacity = District_Median × Random_Variation(0.85, 1.15)
```

**Validation**: Plausibility checks (10-200 spots range)

### Final Calibration
All estimates scaled to match Berlin's known total capacity of 200,000 spots.

## 3. Spatial Analysis

### Catchment Area Calculation
- **Isochrone Generation**: 500m walking-distance polygons using OpenRouteService API
- **Overlap Removal**: Geometric operations to prevent double-counting
- **API Rate Limiting**: Respects service limits (450 requests/session, 11 requests/minute)

### Accessibility Assessment
- **Walking Distance**: Pedestrian-focused routing calculations
- **Service Coverage**: District-level aggregation of reachable areas
- **Environmental Integration**: Proximity to green spaces and water bodies

## 4. Forecasting & Analysis

### Demographic Forecasting
- **Prophet Models**: Facebook's time series forecasting for population trends
- **Exponential Smoothing**: Alternative trend analysis approach
- **Demand Projection**: Translation of population growth to daycare needs

### Coverage Assessment
- **Supply-Demand Ratio**: Current capacity vs. demographic requirements
- **Gap Analysis**: Identification of underserved areas
- **Future Scenarios**: Projected needs until 2034

## 5. Validation & Quality Assurance

### Data Quality Checks
- **Completeness**: Missing value analysis and imputation strategies
- **Spatial Accuracy**: Coordinate validation and outlier detection
- **Temporal Consistency**: Cross-validation with official statistics

### Statistical Validation
- **Model Performance**: Regression diagnostics and residual analysis
- **Uncertainty Quantification**: Confidence intervals for predictions
- **Sensitivity Analysis**: Parameter robustness testing

## 6. Limitations

### Data Constraints
- **OSM Completeness**: Variable data quality across districts
- **Capacity Information**: Limited availability requiring estimation
- **Temporal Snapshot**: Point-in-time analysis with forecast uncertainty

### Methodological Assumptions
- **Linear Relationships**: Simplified capacity-area modeling
- **Static Demand**: No dynamic demand modeling
- **Administrative Focus**: District-level aggregation may mask local variations

---

**For detailed implementation**, see the notebooks in `notebooks/`.
**For code documentation**, review `src/spatial_analysis.py` and `src/config.py`.