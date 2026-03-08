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
# # 01 — Daycare Data Processing
# > ETL from OpenStreetMap: load, clean, estimate missing capacities, export.
#
# ---

# %% [markdown]
# ## Setup

# %%
from pathlib import Path
import sys

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm

sys.path.insert(0, str(Path("../src").resolve()))

from config import (
    DAYCARE_OSM_FILE as GEOJSON_PATH,
    DAYCARE_PROCESSED_FILE as OUTPUT_FILE,
    PROJ_CRS,
    TARGET_CAPACITY,
    WFS_URL,
)

# %% [markdown]
# ## 1. Data Loading
#
# Load the raw daycare GeoJSON and the Berlin district boundaries from the WFS endpoint.
# The preview confirms that both inputs are available and structurally plausible before processing.

# %%
gdf = gpd.read_file(GEOJSON_PATH)
berlin_districts = gpd.read_file(WFS_URL)

print("\nDaycare data:")
display(gdf.head())
print("\nDistrict data:")
display(berlin_districts.head())

# %% [markdown]
# ### Cleaning and spatial assignment
#
# Convert the raw inputs to a common projected CRS, standardize missing values, and attach each daycare center to its district.
# The quick map provides a visual sanity check before capacity estimation.

# %%
if gdf.crs.is_geographic:
    gdf = gdf.to_crs(PROJ_CRS)
berlin_districts = berlin_districts.to_crs(gdf.crs)

gdf = gdf.replace({None: np.nan})
gdf['name'] = gdf['name'].fillna("unknown")
gdf['capacity'] = pd.to_numeric(gdf['capacity'], errors='coerce')

gdf = gpd.sjoin(
    gdf,
    berlin_districts[['geometry', 'namgem']], 
    how="left",
    predicate="within"
)

gdf['suburb'] = gdf['namgem']
gdf = gdf.drop(columns=['namgem', 'index_right'])

fig, ax = plt.subplots(figsize=(12, 8))
berlin_districts.plot(ax=ax, edgecolor='black', facecolor='none')
gdf.plot(ax=ax, color='red', markersize=1)
plt.title('Kita-Standorte in Berlin')
plt.axis('off')
plt.show()

# %% [markdown]
# ### Geometry split and known capacities
#
# Separate point features from polygon features because they need different estimation strategies.
# The histogram of known capacities serves as a baseline check before prediction.

# %%
nodes = gdf[gdf.geometry.type == 'Point'].copy()
polygons = gdf[gdf.geometry.type == 'Polygon'].copy()

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
# ### Polygon-based capacity estimation
#
# Fit a simple regression from building area to capacity for polygon daycare features with known values.
# Use that model to estimate missing capacities and cap extreme predictions with a basic outlier rule.

# %%
valid_data = polygons[polygons['capacity'].notna() & polygons['area'].notna()]

plt.figure(figsize=(10, 6))
plt.scatter(valid_data['area'], valid_data['capacity'])
plt.xlabel('Area (m²)') 
plt.ylabel('Capacity')
plt.title('Relationship between Area and Capacity')
plt.show()

X = valid_data['area'].values.reshape(-1, 1)
X = sm.add_constant(X)
y = valid_data['capacity'].values
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
# ### Point-based capacity estimation
#
# For daycare points without known capacities, use district medians with a small random variation and a plausible min/max clamp.
# This keeps the estimation strategy simple while reflecting local differences in daycare size.

# %%
# Calculate district-wise medians
known_capacities = pd.concat([
    nodes[nodes['capacity'].notna()][['capacity', 'suburb']],
    polygons[['capacity', 'suburb']]
])
global_median = known_capacities['capacity'].median()
district_medians = known_capacities.groupby('suburb')['capacity'].median()

nodes_missing = nodes[nodes['capacity'].isna()].copy()
nodes_missing['predicted_capacity'] = np.nan

for suburb in nodes_missing['suburb'].unique():
    mask = nodes_missing['suburb'] == suburb
    n_samples = mask.sum()

    base_capacity = district_medians.get(suburb, global_median)
    variation = np.random.uniform(0.85, 1.15, n_samples)
    capacities = (base_capacity * variation).round()
    capacities = np.clip(capacities, 10, 200)

    nodes_missing.loc[mask, 'predicted_capacity'] = capacities

nodes.loc[nodes['capacity'].isna(), 'predicted_capacity'] = nodes_missing['predicted_capacity']

plt.figure(figsize=(10, 6))
sns.boxplot(x='suburb', y='predicted_capacity', data=nodes_missing)
plt.xticks(rotation=90)
plt.title('Estimated Capacities by District')
plt.xlabel('District')
plt.ylabel('Estimated Capacity')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data=missing_capacity, x='predicted_capacity', hue='is_outlier', multiple="stack")
plt.title('Distribution of Predicted Capacities')
plt.show()

# %% [markdown]
# ### Scaling to the target capacity
#
# Scale the estimated values so the combined dataset matches the known Berlin-wide target capacity.
# This preserves the relative structure of the predictions while calibrating the total.

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
# ## Results / Summary
#
# Merge the point and polygon datasets into one final daycare layer with observed and predicted capacities.
# The summary statistics and plots provide a final plausibility check before export.

# %%
columns = ['name', 'id', 'capacity', 'predicted_capacity', 'geometry', 'suburb']
combined = pd.concat([nodes[columns], polygons[columns]], ignore_index=True)

combined['final_capacity'] = combined['capacity'].fillna(combined['predicted_capacity'])
combined['capacity_source'] = np.where(
    combined['capacity'].isna(),
    'predicted',
    'observed'
)

# %%
print("\nFinal statistics:")
print(f"Total daycare centers: {len(combined):,}")
print(f"Estimated capacities: {(combined['capacity_source'] == 'predicted').sum():,}")
print(f"Total capacity: {combined['final_capacity'].sum():,.0f}")

district_stats = combined.groupby('suburb').agg({
    'final_capacity': ['sum', 'mean', 'count']
}).round(0)
display(district_stats)


# %% [markdown]
# ## Export
#
# Final processed dataset written to `data/processed/daycare_centers_processed.geojson`.

# %%
final_columns = ['name', 'id', 'suburb', 'final_capacity', 'capacity_source', 'geometry']
combined = combined[final_columns]

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
combined.to_file(OUTPUT_FILE, driver='GeoJSON')
print(f"\nData saved to: {OUTPUT_FILE}")

# %%
# Create visualization layout with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Create boxplot showing capacity distribution by data source (observed vs predicted)
sns.boxplot(x='capacity_source', y='final_capacity', data=combined, ax=ax1)
ax1.set_title('Capacity Distribution by Source')

# Calculate and plot total capacity per district as bar chart
district_stats = combined.groupby('suburb')['final_capacity'].sum().sort_values()
district_stats.plot(kind='bar', ax=ax2)
ax2.set_title('Total Capacity per District')
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# Additional analysis: Create stacked histogram showing capacity distribution by source
plt.figure(figsize=(10, 6))
sns.histplot(data=combined, x='final_capacity', hue='capacity_source', multiple="stack")
plt.title('Distribution of Capacities by Data Source')
plt.show()
