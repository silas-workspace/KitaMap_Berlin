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
# # 03 — District Analysis
# > District-level coverage assessment: supply rates, accessibility, trend categorization, CARTO export.
#
# ---

# %% [markdown]
# ## Setup

# %%
from pathlib import Path
import sys

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import display

sys.path.insert(0, str(Path("../src").resolve()))

from config import (
    DAYCARE_PROCESSED_FILE as DAYCARE_INPUT,
    GREEN_AREAS_FILE as NATURE_INPUT,
    ISOCHRONES_FILE as ISOCHRONE_INPUT,
    POPULATION_FORECAST_FILE as FORECAST_INPUT,
    PROJ_CRS as PROJECT_CRS,
    WATER_AREAS_FILE as WATER_INPUT,
    WFS_URL,
)

# %% [markdown]
# ## Data Import and Initial Review
#
# In this section, the relevant datasets from various sources are imported to create a solid foundation for further analyses:
#
# 1. **District Data**  
#    - The geodata of Berlin's districts are imported via a WFS (Web Feature Service).  
#    - A brief preview shows the first rows of the data and lists the available columns to provide an overview of the structure of the geodata.
#
# 2. **Daycare Data**  
#    - Next, the daycare locations are imported as geodata.  
#    - In addition to a sample review (head), the total number of daycare locations is displayed.  
#    - The column overview clarifies which information is available for each daycare center.
#
# 3. **Forecast Data**  
#    - Finally, the forecasted child numbers (up to 2039) are loaded and transposed to obtain a suitable table structure for further evaluations.  
#    - A look at the columns shows how the forecast values for the individual districts are stored.
#
# This initial review helps to assess data quality and get an impression of the scope and structure of the datasets.
#

# %%
# Load district data
print("District data:")
berlin_districts = gpd.read_file(WFS_URL)
display(berlin_districts.head(2))
print("\nColumns:", berlin_districts.columns.tolist())

# Load daycare data
print("\nDaycare data:")
daycare_centers = gpd.read_file(DAYCARE_INPUT)
display(daycare_centers.head(2))
print(f"\nNumber of daycare locations: {len(daycare_centers)}")
print("Columns:", daycare_centers.columns.tolist())

# Load forecast data
print("\nForecast data:")
forecast = pd.read_csv(FORECAST_INPUT, index_col=0)
forecast_transposed = forecast.T
display(forecast_transposed.head(2))
print("\nColumns:", forecast_transposed.columns.tolist())

# %% [markdown]
# ## Calculation of Summary Statistics per District
#
# In this section, the imported daycare data are grouped by district and basic statistics are determined:
#
# 1. **Grouping and Aggregation**  
#    - Daycare centers are grouped by the *suburb* (district) column.  
#    - For each district, the **number** of daycare centers and the **sum** of daycare capacities are calculated.
#
# 2. **Data Preparation**  
#    - For better readability, columns are renamed (e.g., *suburb* → *District*).  
#    - Capacity values are converted to integers to ensure uniform representation.
#
# 3. **Cleaning and Sorting**  
#    - Entries with the district *Unknown* are excluded.  
#    - The result is sorted in descending order by the number of daycare centers to quickly identify districts with particularly many daycare centers.
#
# These aggregated statistics provide an initial overview of how daycare centers are distributed across Berlin and which districts have particularly high or low childcare capacities.
#

# %%
# Grouping and calculating statistics
daycare_stats = (daycare_centers.groupby('suburb')
            .agg({
                'name': 'count',  # Number of daycare centers per district
                'final_capacity': 'sum'  # Sum of capacities per district
            })
            .reset_index())

# Renaming columns
daycare_stats = daycare_stats.rename(columns={
    'suburb': 'district', 
    'name': 'num_daycares',
    'final_capacity': 'total_capacity'
})

# Integer capacity
daycare_stats['total_capacity'] = daycare_stats['total_capacity'].astype(int)

# Removing "Unknown" and sorting by number of daycare centers
daycare_stats = (daycare_stats[daycare_stats['district'] != 'Unknown']
                .sort_values('num_daycares', ascending=False))

# Preview of the calculated statistics
print("Calculated daycare center statistics per district:")
display(daycare_stats)

# %% [markdown]
# ## Visualization of Daycare Statistics per District
#
# In this step, the district data are examined based on two characteristics:
#
# 1. **Sorting by Number of Daycare Centers**  
#    - First, the table is sorted by the *Number of Daycare Centers* column in ascending order. 
#    - Then, a bar chart shows which districts have the fewest and which have the most daycare centers.
#
# 2. **Sorting by Total Capacity**  
#    - Additionally, the table is sorted by *Total Capacity* to identify districts with low or high numbers of childcare places.
#    - Here too, a bar chart illustrates the distribution of total capacities between the districts.
#
# The combination of both representations makes it clear in which districts there are many small or few large daycare centers and where better coverage of demand may be required.
#

# %%
# Sorting data by 'Number of Daycare Centers' and 'Total Capacity' in ascending order
daycare_stats_sorted_centers = daycare_stats.sort_values(by='num_daycares', ascending=True)
daycare_stats_sorted_capacity = daycare_stats.sort_values(by='total_capacity', ascending=True)

# Creating the figure with more height to accommodate labels
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# Number of daycare centers per district (sorted)
sns.barplot(
    data=daycare_stats_sorted_centers, 
    x='district', 
    y='num_daycares', 
    hue='district',
    dodge=False,
    ax=ax1, 
    palette=sns.color_palette("Blues", n_colors=len(daycare_stats_sorted_centers['district'].unique())),
    errorbar=None,
    legend=False
)
ax1.set_title('Number of Daycare Centers per District', fontsize=16, pad=20)
ax1.set_xlabel('District', fontsize=14)
ax1.set_ylabel('Number of Centers', fontsize=14)

# Rotate labels and adjust their position
ax1.tick_params(axis='x', labelrotation=90, labelsize=10)  # Vertical labels
ax1.tick_params(axis='y', labelsize=10)

# Total capacity per district (sorted)
sns.barplot(
    data=daycare_stats_sorted_capacity, 
    x='district', 
    y='total_capacity', 
    hue='district',
    dodge=False,
    ax=ax2, 
    palette=sns.color_palette("Reds", n_colors=len(daycare_stats_sorted_capacity['district'].unique())),
    errorbar=None,
    legend=False
)
ax2.set_title('Total Capacity per District', fontsize=16, pad=20)
ax2.set_xlabel('District', fontsize=14)
ax2.set_ylabel('Total Capacity', fontsize=14)

# Rotate labels and adjust their position
ax2.tick_params(axis='x', labelrotation=90, labelsize=10)  # Vertical labels
ax2.tick_params(axis='y', labelsize=10)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Add more bottom margin
plt.subplots_adjust(bottom=0.2)

plt.show()

# %% [markdown]
# ## Calculation of Supply Rates and Coverage per District
#
# This section calculates the supply rates for daycare places in each district based on forecasted child numbers and analyzes the spatial coverage:
#
# 1. **Supply Rate Calculation**  
#    - The supply rate is determined by dividing the total daycare capacity by the forecasted number of children in each district for the target years.
#    - This provides an estimate of how well the existing places meet future demand.
#
# 2. **Spatial Coverage Analysis**  
#    - District geometries are overlaid with isochrones, nature, and water areas to determine the percentage of each district that is covered by accessible daycare places.
#    - The analysis highlights differences in accessibility and spatial characteristics between districts.
#
# These calculations help identify districts with potential undersupply or areas where accessibility may be limited due to natural or urban features.
#

# %%
# Merge daycare statistics with district data
districts_with_stats = gpd.GeoDataFrame(
    berlin_districts.merge(
        daycare_stats,
        left_on='namgem', 
        right_on='district',
        how='left'
    ),
    geometry='geometry',
    crs=PROJECT_CRS
)

# Prepare forecast data
# Rename columns from years to more descriptive names
forecast_renamed = forecast.rename(columns={
    '2024': 'forecast_2024',
    '2029': 'forecast_2029',
    '2034': 'forecast_2034'
})

# Final merge of all data
# Convert the result back to GeoDataFrame after merge
districts_with_all = gpd.GeoDataFrame(
    districts_with_stats.merge(
        forecast_renamed,
        left_on='namgem',
        right_index=True,
        how='left'
    ),
    geometry='geometry',
    crs=PROJECT_CRS
)

# Select relevant columns and reorder
final_districts = gpd.GeoDataFrame(
    districts_with_all[[
        'namgem',
        'num_daycares',
        'total_capacity',
        'forecast_2024',
        'forecast_2029',
        'forecast_2034',
        'geometry'
    ]].rename(columns={'namgem': 'district'}),
    geometry='geometry',
    crs=PROJECT_CRS
)

# Output the data
print("\nOverview of merged data:")
display(final_districts[['district', 'num_daycares', 'total_capacity', 'forecast_2024', 'forecast_2029']].head())

print("\nAvailable columns:")
for col in final_districts.columns:
    print(f"- {col}")

# %% [markdown]
# ## Categorization and Trend Analysis
#
# In this section, districts are categorized based on their supply rates and trends over time are analyzed:
#
# 1. **Categorization**  
#    - Districts are classified into categories such as *optimally supplied*, *well supplied*, or *undersupplied* based on their supply rates.
#    - This helps to quickly identify areas with particularly good or poor childcare provision.
#
# 2. **Trend Analysis**  
#    - Changes in supply rates between 2024 and 2039 are examined to highlight districts with expected improvement or deterioration.
#    - The analysis provides insights into future developments and planning needs.
#
# This categorization and trend analysis support targeted planning and resource allocation for daycare provision in Berlin.
#

# %%
# Load additional geospatial data
isochrones = gpd.read_file(ISOCHRONE_INPUT)
nature_area = gpd.read_file(NATURE_INPUT)
water_area = gpd.read_file(WATER_INPUT)
    
# Create a GeoSeries with CRS for all geometries
all_geometries = gpd.GeoSeries(
    list(isochrones.geometry) + 
    list(nature_area.geometry) + 
    list(water_area.geometry),
    crs=isochrones.crs
)

# Union as GeoDataFrame
united_gdf = gpd.GeoDataFrame(geometry=[all_geometries.union_all()], crs=isochrones.crs)

# CRS transformation if necessary
if final_districts.crs != united_gdf.crs:
    united_gdf = united_gdf.to_crs(final_districts.crs)

# List for the results
coverage_percentages = []

# Calculation for each district
for idx, row in final_districts.iterrows():
    # Intersection between district and isochrones
    intersection = row.geometry.intersection(united_gdf.geometry.iloc[0])
    # Calculation of the percentage share
    coverage_percentage = (intersection.area / row.geometry.area) * 100
    coverage_percentages.append(coverage_percentage)

# Add coverage to the DataFrame
final_districts['coverage_pct'] = coverage_percentages
final_districts['coverage_pct'] = final_districts['coverage_pct'].round(2)

print("\nCoverage by district (%):")
display(final_districts[['district', 'coverage_pct']].sort_values('coverage_pct', ascending=False))

# %% [markdown]
# ### Supply categories and trends
#
# Compute supply rates for the target years and classify districts into broad supply categories using the same threshold logic already present in the notebook.
# The resulting trend labels summarize whether district-level coverage improves, declines, or remains stable over time.

# %%
# NOTE: Column names in the committed GeoJSON files (data/external/) are in German,
# matching the CARTO dashboard configuration. Re-running this cell will produce
# English column names. Update CARTO layer configurations before doing so.
for year in [2024, 2029, 2034]:
    final_districts[f'supply_rate_{year}'] = (
        final_districts['total_capacity'] / final_districts[f'forecast_{year}'] * 100
    ).round(2)

category_mapping = {
    'Optimal': lambda row: row['supply_rate_2024'] >= 95 and row['coverage_pct'] >= 80,
    'Good': lambda row: row['supply_rate_2024'] >= 85 and row['coverage_pct'] >= 70,
    'Insufficient': lambda row: row['supply_rate_2024'] >= 75 and row['coverage_pct'] >= 60,
    'Under-supplied': lambda row: row['supply_rate_2024'] >= 65,
    'Critical': lambda row: True  # Default case
}

category_mapping_2034 = {
    'Optimal': lambda row: row['supply_rate_2034'] >= 95 and row['coverage_pct'] >= 80,
    'Good': lambda row: row['supply_rate_2034'] >= 85 and row['coverage_pct'] >= 70,
    'Insufficient': lambda row: row['supply_rate_2034'] >= 75 and row['coverage_pct'] >= 60,
    'Under-supplied': lambda row: row['supply_rate_2034'] >= 65,
    'Critical': lambda row: True  # Default case
}

# Categorization for 2024
final_districts['category_2024'] = final_districts.apply(
    lambda row: next(
        category for category, condition in category_mapping.items() 
        if condition(row)
    ),
    axis=1
)

# Categorization for 2034
final_districts['category_2034'] = final_districts.apply(
    lambda row: next(
        category for category, condition in category_mapping_2034.items() 
        if condition(row)
    ),
    axis=1
)

# Trend calculation (Change 2024 to 2034)
final_districts['change_2024_2034'] = (
    final_districts['supply_rate_2034'] - final_districts['supply_rate_2024']
).round(2)

# Trend categorization with German labels
trend_mapping = {
    'Improving': lambda x: x > 5,
    'Declining': lambda x: x < -5,
    'Stable': lambda x: True  # Default case
}

# Trend categorization
final_districts['trend_2024_2034'] = final_districts['change_2024_2034'].apply(
    lambda x: next(
        trend for trend, condition in trend_mapping.items()
        if condition(x)
    )
)

print("\nDistrict supply overview:")
display(final_districts[[
    'district', 
    'supply_rate_2024',
    'coverage_pct',
    'category_2024',
    'change_2024_2034',
    'trend_2024_2034',
    'category_2034',
]].sort_values('supply_rate_2024', ascending=False))

# %% [markdown]
# ## Data Output and Visualization
#
# In the final step, the processed data are exported and visualized:
#
# 1. **Data Export**  
#    - The final results are saved in GeoJSON format for further use in GIS applications and mapping tools.
#
# 2. **Visualization**  
#    - Charts and maps are created to illustrate the supply situation and spatial distribution of daycare places in Berlin.
#    - The visualizations provide a clear overview of the results and support communication with stakeholders and decision-makers.
#
# This output and visualization step ensures that the analysis results are accessible and usable for further planning and presentation.
#

# %%
# NOTE: Column names in the committed GeoJSON files (data/external/) are in German,
# matching the CARTO dashboard configuration. Re-running this cell will produce
# English column names. Update CARTO layer configurations before doing so.
output_dir = Path("../data/external")
output_files = {
    'base': {
        'name': 'daycare_coverage_base.geojson',
        'columns': [
            'district', 'geometry', 'num_daycares', 'total_capacity',
            'coverage_pct', 'forecast_2024', 'supply_rate_2024'
        ]
    },
    'category': {
        'name': 'daycare_coverage_category_2024.geojson',
        'columns': ['district', 'geometry', 'category_2024']
    },
    'trend': {
        'name': 'daycare_coverage_trend_2024_2034.geojson',
        'columns': ['district', 'geometry', 'trend_2024_2034']
    },
    'category_2034': {
        'name': 'daycare_coverage_category_2034.geojson',
        'columns': ['district', 'geometry', 'category_2034']
    }
}

# Create output directory
output_dir.mkdir(parents=True, exist_ok=True)

# Process and save each file
for key, file_info in output_files.items():
    # Create GeoDataFrame with selected columns
    gdf = gpd.GeoDataFrame(
        final_districts[file_info['columns']]
    ).to_crs('EPSG:4326')
    
    # Define output path
    output_path = output_dir / file_info['name']
    
    # Save to GeoJSON
    gdf.to_file(output_path, driver='GeoJSON')
    
    # Verify file creation and print info
    if output_path.exists():
        file_size = output_path.stat().st_size / 1024  # Convert to KB
        print(f"{file_info['name']} created successfully")
        print(f"   Size: {file_size:.1f} KB")
        print(f"   Number of features: {len(gdf)}")
        print(f"   Columns: {', '.join(gdf.columns)}\n")
    else:
        print(f"Error creating {file_info['name']}")
