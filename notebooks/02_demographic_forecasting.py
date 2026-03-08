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
# # 02 — Demographic Forecasting
# > Population forecast for children aged 0–6 by district (2024–2034) using ETS, Prophet, and Ensemble.
#
# ---

# %% [markdown]
# ## Setup

# %%
from math import sqrt
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

sys.path.insert(0, str(Path("../src").resolve()))

from config import POPULATION_FILE as INPUT_PATH, POPULATION_FORECAST_FILE as OUTPUT_PATH

# %% [markdown]
# ## 1. Data Loading
#
# Load the historical district-level population series for children aged 0–6 and prepare the forecast frame.
# The initial chart provides a quick sanity check on the observed trends before modeling.

# %%
df = pd.read_csv(INPUT_PATH)
year_columns = [str(year) for year in range(2015, 2025)]
df = df[['Bezirk'] + year_columns]

plt.figure(figsize=(15, 8))
sns.set_style("whitegrid")

colors = sns.color_palette("husl", n_colors=len(df))

for idx, row in df.iterrows():
    district = row['Bezirk']
    values = row[year_columns].values
    plt.plot(year_columns, values, marker='o', label=district, color=colors[idx], linewidth=2)

plt.title('Development of Child Numbers 2015-2024', fontsize=14, pad=20)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Number of Children', fontsize=12)
plt.xticks(rotation=45)

plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))

plt.legend(
    bbox_to_anchor=(1.05, 1),
    loc='upper left',
    borderaxespad=0.0,
    title='Districts',
    frameon=True,
)

plt.tight_layout()
plt.show()

# Create results DataFrame with districts as index and forecast years as columns
forecast_years = [2024, 2029, 2034]
results = pd.DataFrame(index=df['Bezirk'], columns=forecast_years)

# Fill in 2024 values
results[2024] = df['2024'].values

print("\nResults DataFrame Structure:")
print(results)


# %% [markdown]
# ### Error metrics and evaluation helpers
#
# Define a few standard forecast error metrics and a shared evaluation helper.
# These values are reused for ETS, Prophet, and the ensemble comparison.

# %%
def calculate_rmse(y_true, y_pred):
    return sqrt(mean_squared_error(y_true, y_pred))


def calculate_mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)


def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def evaluate_model(model_name, y_true, y_pred, bezirk):
    return {
        'Model': model_name,
        'Bezirk': bezirk,
        'RMSE': calculate_rmse(y_true, y_pred),
        'MAE': calculate_mae(y_true, y_pred),
        'MAPE': calculate_mape(y_true, y_pred)
    }


# %%
# Define forecast years
forecast_years = [2029, 2034]

# Initialize DataFrame for all predictions
all_predictions = pd.DataFrame(
    index=df['Bezirk'],
    columns=[
        f'{model}_{year}' 
        for model in ['ETS', 'Prophet', 'Ensemble'] 
        for year in forecast_years
    ]
)

evaluation_results = []

# %% [markdown]
# ### Exponential Smoothing
#
# Fit an ETS model for each district, store the forecast years, and evaluate the holdout fit on the last two observed values.

# %%
# Exponential Smoothing for each district
for district in df['Bezirk']:
    # Prepare historical data - now in correct chronological order
    historical_values = df[df['Bezirk'] == district][year_columns].values.flatten()
    
    # Train-Test-Split for evaluation
    train_data = historical_values[:-2]
    test_data = historical_values[-2:]
    
    # Exponential Smoothing forecast
    exp_model = ExponentialSmoothing(
        historical_values, trend='add', seasonal=None, damped_trend=True
    ).fit()
    exp_forecast = exp_model.forecast(16)
    
    # Save the ETS forecasts
    for i, year in enumerate(forecast_years):
        all_predictions.loc[district, f'ETS_{year}'] = np.round(exp_forecast[4 + i*5], 0)
    
    # Evaluation
    exp_model_eval = ExponentialSmoothing(
        train_data, trend='add', seasonal=None, damped_trend=True
    ).fit()
    exp_pred_eval = exp_model_eval.forecast(2)
    evaluation_results.append(
        evaluate_model('Exponential Smoothing', test_data, exp_pred_eval, district)
    )

# Display results for ETS
print("\nExponential Smoothing forecasts:")
print(all_predictions.filter(like='ETS'))
print("\nETS Evaluation:")
ets_eval = pd.DataFrame(evaluation_results)
print(ets_eval.groupby('Model')[['RMSE', 'MAE', 'MAPE']].mean())

# %% [markdown]
# ### Prophet
#
# Fit a Prophet model per district and evaluate it on the same short holdout window used for ETS.
# This keeps the comparison consistent across forecasting approaches.

# %%
# Prophet for each district
for district in df['Bezirk']:
    # Extract historical data - and flatten() to make it 1-dimensional
    historical_values = df[df['Bezirk'] == district][year_columns].values.flatten()
    
    # Create date range
    dates = pd.date_range(start='2015', periods=len(year_columns), freq='Y')
    
    # Prepare data for Prophet
    prophet_df = pd.DataFrame({
        'ds': dates,
        'y': historical_values
    })
    
    # Fit model and make forecast
    prophet_model = Prophet(yearly_seasonality=False)
    prophet_model.fit(prophet_df)
    
    # Create future data
    future = pd.DataFrame({
        'ds': [pd.Timestamp(str(year)) for year in forecast_years]
    })
    prophet_forecast = prophet_model.predict(future)
    
    # Extract forecasts for the desired years
    for year in forecast_years:
        forecast_value = prophet_forecast[prophet_forecast['ds'].dt.year == year]['yhat'].values[0]
        all_predictions.loc[district, f'Prophet_{year}'] = np.round(forecast_value, 0)
    
    # Evaluation
    train_data = historical_values[:-2]
    test_data = historical_values[-2:]
    
    # Prepare evaluation dataset
    train_dates = dates[:-2]
    prophet_df_eval = pd.DataFrame({
        'ds': train_dates,
        'y': train_data
    })
    
    # Fit evaluation model
    prophet_model_eval = Prophet(yearly_seasonality=False)
    prophet_model_eval.fit(prophet_df_eval)
    
    # Forecast for evaluation period
    future_eval = pd.DataFrame({'ds': dates[-2:]})
    prophet_pred_eval = prophet_model_eval.predict(future_eval)['yhat'].values
    
    evaluation_results.append(
        evaluate_model('Prophet', test_data, prophet_pred_eval, district)
    )

# Display results for Prophet
print("\nProphet forecasts:")
print(all_predictions.filter(like='Prophet'))
print("\nProphet Evaluation:")
prophet_eval = pd.DataFrame(evaluation_results)
print(prophet_eval[prophet_eval['Model'] == 'Prophet'].groupby('Model')[['RMSE', 'MAE', 'MAPE']].mean())

# %% [markdown]
# ### Ensemble
#
# Combine ETS and Prophet forecasts with weights derived from their mean MAPE.
# The ensemble serves as the final forecast written to the output table.

# %%
# Method 3: Ensemble

# Calculate weights based on the performance (MAPE) of the models
ets_mape = ets_eval[ets_eval['Model'] == 'Exponential Smoothing']['MAPE'].mean()
prophet_mape = prophet_eval[prophet_eval['Model'] == 'Prophet']['MAPE'].mean()

# Use inverse MAPE for weighting
ets_weight = (1/ets_mape) / (1/ets_mape + 1/prophet_mape)
prophet_weight = (1/prophet_mape) / (1/ets_mape + 1/prophet_mape)

# Ensemble for each district
for district in df['Bezirk']:
    # Extract historical data in correct order
    historical_values = df[df['Bezirk'] == district][year_columns].values.flatten()
    dates = pd.date_range(start='2015', periods=len(year_columns), freq='Y')
    
    # Calculate ensemble forecast and save
    for year in forecast_years:
        ets_pred = all_predictions.loc[district, f'ETS_{year}']
        prophet_pred = all_predictions.loc[district, f'Prophet_{year}']
        ensemble_pred = (ets_weight * ets_pred + prophet_weight * prophet_pred)
        all_predictions.loc[district, f'Ensemble_{year}'] = np.round(ensemble_pred, 0)
    
    # Evaluation
    train_data = historical_values[:-2]
    test_data = historical_values[-2:]
    
    # ETS Evaluation
    exp_model_eval = ExponentialSmoothing(
        train_data, trend='add', seasonal=None, damped_trend=True
    ).fit()
    ets_pred_eval = exp_model_eval.forecast(2)
    
    # Prophet Evaluation
    train_dates = dates[:-2]
    prophet_df_eval = pd.DataFrame({
        'ds': train_dates,
        'y': train_data
    })
    prophet_model_eval = Prophet(yearly_seasonality=False)
    prophet_model_eval.fit(prophet_df_eval)
    future_eval = pd.DataFrame({'ds': dates[-2:]})
    prophet_pred_eval = prophet_model_eval.predict(future_eval)['yhat'].values
    
    # Ensemble Evaluation
    ensemble_pred_eval = ets_weight * ets_pred_eval + prophet_weight * prophet_pred_eval
    
    evaluation_results.append(
        evaluate_model('Ensemble', test_data, ensemble_pred_eval, district)
    )

# Display results for Ensemble
print("\nEnsemble forecasts:")
print(all_predictions.filter(like='Ensemble'))
print("\nEnsemble Evaluation:")
ensemble_eval = pd.DataFrame(evaluation_results)
print(ensemble_eval[ensemble_eval['Model'] == 'Ensemble'].groupby('Model')[['RMSE', 'MAE', 'MAPE']].mean())

# Transfer final Ensemble forecasts to results DataFrame
for year in forecast_years:
    if year == 2024:
        continue  # 2024 we already have from real data
    results[year] = all_predictions[f'Ensemble_{year}']

# %% [markdown]
# ## Results / Summary
#
# Review the aggregated model metrics and compare the best-performing model per district.
# The ensemble output is then prepared for export.

# %%
# Convert Evaluation Results to DataFrame
eval_df = pd.DataFrame(evaluation_results)

# Average metrics per model
print("Average metrics per model:")
print(eval_df.groupby('Model')[['RMSE', 'MAE', 'MAPE']].mean())

# Best models per district based on MAPE
print("\nBest models per district (based on MAPE):")
best_models = eval_df.loc[eval_df.groupby('Bezirk')['MAPE'].idxmin()]
print(best_models[['Bezirk', 'Model', 'MAPE']])

# %% [markdown]
# ## Export
#
# Final district forecasts are written to `data/processed/population_forecast_2024_2034.csv`.

# %%
results = results.astype(int)

# Save results as CSV
results.to_csv(OUTPUT_PATH)

# Output the save path
print(f"\nForecast saved in: {OUTPUT_PATH}")

plt.figure(figsize=(15, 8))

# Define colors for each district
colors = plt.cm.tab20(np.linspace(0, 1, len(df['Bezirk'])))

# Plot historical data and forecasts for each district
for idx, bezirk in enumerate(df['Bezirk']):
        # Historical data
        historical_values = df[df['Bezirk'] == bezirk].iloc[0, 1:].values
        historical_years = list(range(2015, 2025))
        
        # Forecast values
        forecast_values = results.loc[bezirk]
        all_years = results.columns.astype(int)
        
        # Plot with the same color for historical and forecast
        plt.plot(historical_years, historical_values, 
                marker='o', 
                linestyle='-', 
                color=colors[idx],
                linewidth=2,
                label=f'{bezirk} (Historical)')
        
        plt.plot(all_years, forecast_values, 
                marker='s', 
                linestyle='--', 
                color=colors[idx],
                linewidth=2,
                label=f'{bezirk} (Forecast)')

plt.title('Development and Forecast of Child Numbers 2015-2039', pad=20, size=14)
plt.xlabel('Year', size=12)
plt.ylabel('Number of Children', size=12)

# Set x-axis ticks to show whole years
plt.xticks(range(2015, 2040, 5))  # Show years in 5-year intervals
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))

# Adjust grid
plt.grid(True, linestyle='--', alpha=0.7, color='gray', zorder=0)

# Adjust axis range
plt.ylim(bottom=12000)  # Adjust lower bound

# Optimize legend
plt.legend(bbox_to_anchor=(1.05, 1), 
                loc='upper left', 
                borderaxespad=0,
                frameon=True,
                fancybox=True,
                shadow=True,
                fontsize=10)

# Adjust layout
plt.tight_layout()
plt.show()
