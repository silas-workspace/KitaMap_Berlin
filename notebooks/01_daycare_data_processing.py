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
# # Berlin Daycare Capacity Analysis - Data Processing and Estimation
#
# This notebook is part of the **"KitaMap Berlin"** project and is used for processing and analyzing daycare center data from OpenStreetMap (OSM). The main goal is to create a complete dataset of Berlin daycare centers with their capacities, serving as a basis for further supply analyses.
#
# ## Main Tasks of the Notebook:
#
# 1. **Data Integration**: 
#   - Import OSM daycare data and Berlin district boundaries
#   - Spatially assign daycare centers to districts
#
# 2. **Capacity Estimation**:
#   - Analyze known capacity values
#   - Develop various estimation models for missing capacities:
#     - Area-based regression for polygon geometries
#     - District-specific median estimation for point geometries
#   - Quality assurance through outlier detection and handling
#
# 3. **Data Validation**:
#   - Scale estimates to known total capacity
#   - Statistical analyses and visualizations for plausibility checks
#
# The resulting dataset forms the basis for analyzing the spatial distribution of daycare places and identifying possible supply gaps in Berlin.

# %%
# Import necessary libraries
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from pathlib import Path

# Define constants
GEOJSON_PATH = Path("../data/raw/daycare_centers_osm.geojson")  # Path to the raw daycare data in GeoJSON format
WFS_URL = "https://gdi.berlin.de/services/wfs/alkis_bezirke?REQUEST=GetCapabilities&SERVICE=wfs"  # URL for the Web Feature Service (WFS) for Berlin districts
OUTPUT_FILE = Path("../data/processed/daycare_centers_processed.geojson")  # Path for the processed daycare data output
PROJ_CRS = "EPSG:32633"  # Coordinate Reference System (CRS) for UTM Zone 33N, suitable for Germany
TARGET_CAPACITY = 200_000  # Target capacity for Berlin

# %% [markdown]
# ## Loading the Data Foundations for KitaVision Analysis
#
# In this first step, we load two key datasets that form the basis for our analysis of daycare provision in Berlin:
#
# * **Daycare Centers in Berlin**: From OpenStreetMap, we obtain detailed location data for all recorded daycare centers in Berlin. These data are in GeoJSON format and include important attributes such as names and, if available, capacities of the facilities.
# * **Berlin District Boundaries**: Through a Web Feature Service (WFS), we access the official administrative boundaries of Berlin's districts. These geodata later enable district-specific analysis of daycare provision.
#
# The first rows of both datasets are displayed to gain an initial overview of the available information and to check data quality.
#
#

# %%
# Load Kita data from GeoJSON file
gdf = gpd.read_file(GEOJSON_PATH)
# Load Berlin districts data from Web Feature Service (WFS) URL
berlin_districts = gpd.read_file(WFS_URL)

# Display an overview of the data
print("\nKita Data:")
display(gdf.head())
print("\nDistrict Data:")
display(berlin_districts.head())

# %% [markdown]
# ## Data Preparation and Initial Spatial Visualization
#
# In this phase, we prepare the raw data for further analysis and create an initial cartographic overview:
#
# 1. **Standardization of Geographic Reference Systems**:
# The data are transformed into the UTM coordinate system (Zone 33N). This enables precise distance calculations in meters, which are essential for later accessibility analyses.
#
# 2. **Data Cleaning**:
# - Missing values are uniformly marked as NaN
# - For daycare centers without a name, "unknown" is entered
# - Capacity values are converted to a numeric format
#
# 3. **Spatial Assignment**:
# Through a spatial join, each daycare center is assigned to the corresponding Berlin district. This later enables district-specific evaluations of the supply situation.
#
# 4. **Map Visualization**:
# The final visualization shows the spatial distribution of all daycare locations (red dots) across Berlin's districts. This overview map provides a first impression of the spatial distribution of childcare facilities in the city and already reveals differences in supply density.

# %%
# Transform coordinate reference system (CRS) to UTM Zone 33N
if gdf.crs.is_geographic:
    gdf = gdf.to_crs(PROJ_CRS)
berlin_districts = berlin_districts.to_crs(gdf.crs)

# Basic data cleaning: replace None with NaN, fill missing names, convert capacity to numeric
gdf = gdf.replace({None: np.nan})
gdf['name'] = gdf['name'].fillna("unbekannt")
gdf['capacity'] = pd.to_numeric(gdf['capacity'], errors='coerce')

# Perform spatial join to associate each daycare center with its district
gdf = gpd.sjoin(
    gdf,
    berlin_districts[['geometry', 'namgem']], 
    how="left",
    predicate="within"
)

# Update district information and clean up redundant columns
gdf['suburb'] = gdf['namgem']
gdf = gdf.drop(columns=['namgem', 'index_right'])

# Create visualization of daycare centers on district map
fig, ax = plt.subplots(figsize=(12, 8))
berlin_districts.plot(ax=ax, edgecolor='black', facecolor='none')
gdf.plot(ax=ax, color='red', markersize=1)
plt.title('Kita-Standorte in Berlin')
plt.axis('off')
plt.show()

# %% [markdown]
# ## Analysis of Daycare Geometries and Capacities
#
# The OpenStreetMap data contain two different types of daycare geometries that must be analyzed separately:
#
# 1. **Point Data (Nodes)**:
# These represent daycare centers that are simply recorded as location points. They make up the majority of entries and are typical for smaller facilities.
#
# 2. **Area Data (Polygons)**:
# These show daycare centers where the actual building footprints or property areas have been mapped. This is often the case for larger facilities. For further analysis, these areas are converted to centroids.
#
# The distribution of known capacities is shown in a histogram. This visualization provides important insights into:
# - The typical size ranges of Berlin daycare centers
# - Possible outliers or unusual values
# - The range of care capacities
#
# This information is valuable for later estimation of missing capacity values and for assessing the supply situation.

# %%
# Split data by geometry type into points (nodes) and polygons
nodes = gdf[gdf.geometry.type == 'Point'].copy()
polygons = gdf[gdf.geometry.type == 'Polygon'].copy()

# Calculate area for polygons and convert polygon geometries to centroids
polygons['area'] = polygons.geometry.area.round().astype(float)
polygons['geometry'] = polygons.geometry.centroid

print(f"Nodes: {len(nodes)}, Polygons: {len(polygons)}")

# Create a combined dataset of known capacities from both nodes and polygons
known_capacities = pd.concat([
    nodes[nodes['capacity'].notna()]['capacity'],
    polygons[polygons['capacity'].notna()]['capacity']
])

# Visualize the distribution of known capacities using a histogram
plt.figure(figsize=(10, 6))
sns.histplot(known_capacities, bins=30)
plt.title('Distribution of Known Capacities')
plt.xlabel('Capacity')
plt.ylabel('Count')
plt.show()

# %% [markdown]
# ## Capacity Estimation Based on Building Areas
#
# For daycare centers mapped as areas but lacking capacity information, we develop a statistical model to estimate the number of care places. This approach is based on the assumption that the building area of a daycare center is systematically related to its care capacity.
#
# The analysis is carried out in several steps:
#
# 1. **Correlation Analysis**:
# A scatter plot shows the relationship between building area and known capacities. This provides a first visual impression of the relationship between these variables.
#
# 2. **Regression Model**:
# Using linear regression, the mathematical relationship between area and capacity is modeled. The regression statistics provide information about the quality of the model and the reliability of the estimates.
#
# 3. **Capacity Prediction**:
# For daycare centers without capacity information, care places are estimated based on their area. Confidence intervals are also calculated to quantify the uncertainty of the estimates.
#
# 4. **Outlier Treatment**:
# Unusually high estimated values are identified using statistical methods (IQR method) and replaced with more realistic values. This prevents unrealistic overestimations of capacities.
#
# The final visualization shows the estimated capacities, with potential outliers marked in red. These cleaned estimates complete our dataset for further analysis of the supply situation.

# %%
# Filter data to only include polygons with valid capacity and area values
valid_data = polygons[polygons['capacity'].notna() & polygons['area'].notna()]

# Create scatter plot to visualize relationship between area and capacity
plt.figure(figsize=(10, 6))
plt.scatter(valid_data['area'], valid_data['capacity'])
plt.xlabel('Area (m²)') 
plt.ylabel('Capacity')
plt.title('Relationship between Area and Capacity')
plt.show()

# Perform linear regression analysis
# Reshape area values and add constant term for regression
X = valid_data['area'].values.reshape(-1, 1)
X = sm.add_constant(X)
y = valid_data['capacity'].values
# Fit ordinary least squares regression model
model = sm.OLS(y, X).fit()

print("\nRegression Statistics:")
print(model.summary().tables[1])

# %% [markdown]
# #### Interpretation of the Regression Analysis
#
# The regression analysis shows a significant relationship between building area and daycare capacity:
#
# 1. **Base Capacity** (const = 68.59):
# - On average, a daycare center has a "base capacity" of about 69 places, regardless of its size
# - The value is statistically highly significant (p < 0.001)
#
# 2. **Area Effect** (x1 = 0.0132):
# - For each additional square meter of area, there are on average 0.0132 more care places
# - In other words: For every 100m² of additional area, there are about 1.32 additional care places
# - This relationship is also statistically highly significant (p < 0.001)
#
# The confidence intervals ([0.025 0.975]) show that the true values are between 55 and 82 places for the base capacity and between 0.009 and 0.017 places per m² for the area effect with 95% probability.

# %%
# Predict capacity for polygons with missing values
missing_capacity = polygons[polygons['capacity'].isna()].copy()
X_pred = sm.add_constant(missing_capacity['area'].values.reshape(-1, 1))
missing_capacity['predicted_capacity'] = model.predict(X_pred)

# Transfer predictions back to main polygons DataFrame
polygons.loc[polygons['capacity'].isna(), 'predicted_capacity'] = missing_capacity['predicted_capacity']

# Calculate confidence intervals for predictions
prediction = model.get_prediction(X_pred)
missing_capacity['conf_lower'], missing_capacity['conf_upper'] = prediction.conf_int().T

# Identify outliers using IQR method
Q1 = missing_capacity['predicted_capacity'].quantile(0.25)
Q3 = missing_capacity['predicted_capacity'].quantile(0.75)
IQR = Q3 - Q1
upper_bound = Q3 + 1.5 * IQR
missing_capacity['is_outlier'] = missing_capacity['predicted_capacity'] > upper_bound

# Replace outliers with the median of the non-outliers
median_value = missing_capacity[~missing_capacity['is_outlier']]['predicted_capacity'].median()
missing_capacity.loc[missing_capacity['is_outlier'], 'predicted_capacity'] = median_value


# Visualize predictions highlighting outliers
plt.figure(figsize=(10, 6))
plt.scatter(missing_capacity[~missing_capacity['is_outlier']]['area'], missing_capacity[~missing_capacity['is_outlier']]['predicted_capacity'],label='Normal')
plt.scatter(missing_capacity[missing_capacity['is_outlier']]['area'],missing_capacity[missing_capacity['is_outlier']]['predicted_capacity'],color='red', label='Outliers')
plt.xlabel('Area (m²)')
plt.ylabel('Predicted Capacity')
plt.title('Predicted Capacities with Outliers')
plt.legend()
plt.show()

# %% [markdown]
# ## District-Specific Capacity Estimation for Point Daycare Centers
#
# For daycare centers recorded only as points and lacking capacity information, a district-specific estimation method is developed. This takes local differences in daycare sizes into account:
#
# 1. **Calculation of Reference Values**:
# - For each district, the median of known capacities is determined
# - Additionally, a citywide median is calculated as a fallback
# - These values serve as the basis for the estimates
#
# 2. **Estimation Methodology**:
# - For missing values, the respective district median is used as the basis
# - If no reference values are available for a district, the overall Berlin median is used
# - A random variation of ±15% is added to create realistic dispersion
# - Values are limited to a plausible range of 10 to 200 places
#
# 3. **Visualization of Results**:
# - A boxplot shows the distribution of estimated capacities by district
# - A stacked histogram visualizes the overall distribution of estimated values
# - Outliers are shown separately
#
# This localized estimation method takes district-specific differences in daycare size structure into account and thus produces more realistic capacity estimates than a citywide uniform value.

# %%
# Calculate district-wise medians
known_capacities = pd.concat([
    nodes[nodes['capacity'].notna()][['capacity', 'suburb']],
    polygons[['capacity', 'suburb']]
])
global_median = known_capacities['capacity'].median()  # Calculate the global median
district_medians = known_capacities.groupby('suburb')['capacity'].median()  # Calculate district-wise medians

# Estimate missing capacities
nodes_missing = nodes[nodes['capacity'].isna()].copy()
nodes_missing['predicted_capacity'] = np.nan  # Initialize a column for predicted capacities

for suburb in nodes_missing['suburb'].unique():
    mask = nodes_missing['suburb'] == suburb
    n_samples = mask.sum()  # Count the number of samples for the current suburb
    
    # Determine the base capacity for the current suburb
    base_capacity = district_medians.get(suburb, global_median)
    variation = np.random.uniform(0.85, 1.15, n_samples)  # Generate random variation
    capacities = (base_capacity * variation).round()  # Calculate capacities with variation
    capacities = np.clip(capacities, 10, 200)  # Clip capacities within a reasonable range
    
    nodes_missing.loc[mask, 'predicted_capacity'] = capacities  # Assign predicted capacities

nodes.loc[nodes['capacity'].isna(), 'predicted_capacity'] = nodes_missing['predicted_capacity']  # Transfer predictions back to the main DataFrame

# Visualization of estimated capacities
plt.figure(figsize=(10, 6))
sns.boxplot(x='suburb', y='predicted_capacity', data=nodes_missing)  # Boxplot for visualizing estimated capacities by suburb
plt.xticks(rotation=90)
plt.title('Estimated Capacities by District')
plt.xlabel('District')
plt.ylabel('Estimated Capacity')
plt.show()

# Distribution of predicted capacities
plt.figure(figsize=(10, 6))
sns.histplot(data=missing_capacity, x='predicted_capacity', hue='is_outlier', multiple="stack")  # Histogram for visualizing the distribution of predicted capacities
plt.title('Distribution of Predicted Capacities')
plt.show()

# %% [markdown]
# ## Scaling the Estimated Capacities to Target Value
#
# To ensure that our capacity estimates match the known total number of care places in Berlin, a final adjustment is made:
#
# 1. **Calculation of Total Capacity**:
# - Summing all known and estimated capacities from point and area data
# - This first sum is based on our statistical estimates
#
# 2. **Scaling Factor**:
# - Comparison of the estimated total capacity with the official target capacity
# - Calculation of a scaling factor that adjusts the estimated values to the target value
#
# 3. **Adjustment of Estimates**:
# - All estimated (not the known!) capacity values are multiplied by the scaling factor
# - This ensures that the total number of care places matches reality
# - The relative distribution of the estimated capacities is maintained
#
# This scaling is the last step in capacity estimation and ensures consistency of our data with official statistics.

# %%
# Calculate the current total capacity
total_capacity = (
    nodes['capacity'].fillna(nodes['predicted_capacity']).sum() +
    polygons['capacity'].fillna(polygons['predicted_capacity']).sum()
)

# Adjust to target capacity
scale_factor = TARGET_CAPACITY / total_capacity
print(f"Scale factor: {scale_factor:.3f}")

# Scale the estimated capacities
for gdf in [nodes, polygons]:
    mask = gdf['capacity'].isna()
    gdf.loc[mask, 'predicted_capacity'] = (
        gdf.loc[mask, 'predicted_capacity'] * scale_factor
    ).round()

# %% [markdown]
# ## Finalization and Analysis of the Capacity Dataset
#
# In this final step, the data are merged, analyzed, and prepared for further use:
#
# 1. **Data Merging**:
# - Combining point and area data into a complete dataset
# - Creating a unified capacity column with observed and estimated values
# - Documenting the data source (observed/estimated) for transparency
#
# 2. **Statistical Evaluation**:
# - Overview of Berlin's daycare landscape (number, total capacity)
# - Detailed district statistics with:
#  - Total number of care places
#  - Average daycare size
#  - Number of facilities
#
# 3. **Visualizations**:
# - Boxplots showing the distribution of capacities by data source
# - Bar chart of total capacities by district
# - Stacked histogram visualizing the capacity distribution
#
# 4. **Data Storage**:
# - Creation of a final GeoJSON dataset with all relevant information
# - Saving the data for further analyses and visualizations
#
# This final preparation forms the basis for all further analyses within the KitaVision project and enables in-depth insights into the supply situation.

# %%
# Selection and merging of relevant columns
columns = ['name', 'id', 'capacity', 'predicted_capacity', 'geometry', 'suburb']
combined = pd.concat([nodes[columns], polygons[columns]], ignore_index=True)

# Creation of the final capacity columns
combined['final_capacity'] = combined['capacity'].fillna(combined['predicted_capacity'])
combined['capacity_source'] = np.where(
    combined['capacity'].isna(),
    'predicted',
    'observed'
)

# %%
# Print final statistics
print("\nFinale Statistiken:")
print(f"Gesamtanzahl Kitas: {len(combined):,}")
print(f"Geschätzte Kapazitäten: {(combined['capacity_source'] == 'predicted').sum():,}")
print(f"Gesamtkapazität: {combined['final_capacity'].sum():,.0f}")

# Calculate capacity distribution statistics by district
# Aggregates sum, mean and count of final capacity for each suburb
bezirk_stats = combined.groupby('suburb').agg({
    'final_capacity': ['sum', 'mean', 'count']
}).round(0)
display(bezirk_stats)


# %%
# Select final columns for output dataset
# - name: Name of the daycare center
# - id: Unique identifier
# - suburb: District/neighborhood
# - final_capacity: Final capacity (either observed or predicted)
# - capacity_source: Source of capacity data (observed/predicted)
# - geometry: Geographic location/shape
final_columns = ['name', 'id', 'suburb', 'final_capacity', 'capacity_source', 'geometry']
combined = combined[final_columns]

# Create output directory if it doesn't exist
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# Save final dataset as GeoJSON file
combined.to_file(OUTPUT_FILE, driver='GeoJSON')
print(f"\nData saved to: {OUTPUT_FILE}")

# %%
# Create visualization layout with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Create boxplot showing capacity distribution by data source (observed vs predicted)
sns.boxplot(x='capacity_source', y='final_capacity', data=combined, ax=ax1)
ax1.set_title('Capacity Distribution by Source')

# Calculate and plot total capacity per district as bar chart
bezirk_stats = combined.groupby('suburb')['final_capacity'].sum().sort_values()
bezirk_stats.plot(kind='bar', ax=ax2)
ax2.set_title('Total Capacity per District')
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# Additional analysis: Create stacked histogram showing capacity distribution by source
plt.figure(figsize=(10, 6))
sns.histplot(data=combined, x='final_capacity', hue='capacity_source', multiple="stack")
plt.title('Distribution of Capacities by Data Source')
plt.show()
