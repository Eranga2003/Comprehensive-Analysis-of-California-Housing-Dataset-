# california_housing_analysis.py
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ----- paths
out_dir = Path(".")
fig_dir = out_dir / "figures"
fig_dir.mkdir(exist_ok=True)

# ----- load data (CSV first, then sklearn fallback)
csv_path = Path("housing_data.csv")
if csv_path.exists():
    df = pd.read_csv(csv_path)
    # Normalize common column names to a consistent set
    col_map = {
        'MedInc': 'MedInc', 'median_income': 'MedInc',
        'HouseAge': 'HouseAge', 'housing_median_age': 'HouseAge',
        'AveRooms': 'AveRooms', 'total_rooms': 'TotalRooms',
        'AveBedrms': 'AveBedrms', 'total_bedrooms': 'TotalBedrooms',
        'Population': 'Population', 'population': 'Population',
        'AveOccup': 'AveOccup', 'households': 'Households',
        'Latitude': 'Latitude', 'latitude': 'Latitude',
        'Longitude': 'Longitude', 'longitude': 'Longitude',
        'MedHouseVal': 'MedHouseVal', 'median_house_value': 'MedHouseVal',
        'TotalRooms': 'TotalRooms', 'TotalBedrooms': 'TotalBedrooms',
        'Households': 'Households'
    }
    df = df.rename(columns={c: col_map.get(c, col_map.get(c.lower(), c)) for c in df.columns})
else:
    from sklearn.datasets import fetch_california_housing
    data = fetch_california_housing(as_frame=True)
    df = data.frame.rename(columns={
        'MedInc': 'MedInc', 'HouseAge': 'HouseAge', 'AveRooms': 'AveRooms',
        'AveBedrms': 'AveBedrms', 'Population': 'Population', 'AveOccup': 'AveOccup',
        'Latitude': 'Latitude', 'Longitude': 'Longitude', 'MedHouseVal': 'MedHouseVal'
    })

# ----- engineer totals if missing (best-effort)
if 'TotalRooms' not in df.columns and {'AveRooms', 'Households'}.issubset(df.columns):
    df['TotalRooms'] = df['AveRooms'] * df['Households']
if 'TotalBedrooms' not in df.columns and {'AveBedrms', 'Households'}.issubset(df.columns):
    df['TotalBedrooms'] = df['AveBedrms'] * df['Households']

# ----- keep a copy for reporting (before imputation)
df_before_imp = df.copy()

# ----- impute numeric NaNs with median (so models don’t crash)
for c in df.select_dtypes(include=[np.number]).columns:
    df[c] = df[c].fillna(df[c].median())

# ----- PART B: preview & stats
print("\nFirst 10 rows:")
print(df_before_imp.head(10).to_string(index=False))

desc_cols = [c for c in ['MedInc', 'TotalRooms', 'Population'] if c in df_before_imp.columns]
if desc_cols:
    print("\nDescriptive stats (mean/median/std/min/max):")
    print(df_before_imp[desc_cols].agg(['mean','median','std','min','max']).T)

print("\nMissing values (before imputation):")
print(df_before_imp.isna().sum())

# ----- PART C: plots
if 'MedHouseVal' in df.columns:
    plt.figure()
    df['MedHouseVal'].plot(kind='hist', bins=50)
    plt.xlabel('Median House Value (x $100k if sklearn)')
    plt.ylabel('Count')
    plt.title('Histogram of Median House Value')
    plt.tight_layout()
    plt.savefig(fig_dir / 'figure_1_hist_med_house_value.png')
    plt.close()

if {'MedInc','MedHouseVal'}.issubset(df.columns):
    plt.figure()
    plt.scatter(df['MedInc'], df['MedHouseVal'], alpha=0.3)
    plt.xlabel('Median Income')
    plt.ylabel('Median House Value')
    plt.title('Median Income vs Median House Value')
    plt.tight_layout()
    plt.savefig(fig_dir / 'figure_2_scatter_income_vs_value.png')
    plt.close()

if {'Longitude','Latitude','MedHouseVal'}.issubset(df.columns):
    plt.figure()
    sc = plt.scatter(df['Longitude'], df['Latitude'], c=df['MedHouseVal'], alpha=0.5)
    plt.xlabel('Longitude'); plt.ylabel('Latitude')
    plt.title('Geographical Distribution Colored by Median House Value')
    cbar = plt.colorbar(sc); cbar.set_label('Median House Value')
    plt.tight_layout()
    plt.savefig(fig_dir / 'figure_3_geo_scatter.png')
    plt.close()

# Correlation heatmap
num_df = df.select_dtypes(include=[np.number])
if not num_df.empty:
    corr = num_df.corr(numeric_only=True)
    plt.figure(figsize=(8,6))
    im = plt.imshow(corr, aspect='auto')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.index)), corr.index)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(fig_dir / 'figure_4_correlation_heatmap.png')
    plt.close()

# ----- PART D: models
target = 'MedHouseVal'
features = [c for c in ['MedInc','HouseAge','TotalRooms','TotalBedrooms','Population',
                        'Households','AveRooms','AveBedrms','AveOccup','Latitude','Longitude']
            if c in df.columns and c != target]

metrics = {}

if target in df.columns and features:
    X = df[features].copy()
    y = df[target].copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # Linear Regression
    lr = LinearRegression().fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    mse_lr = mean_squared_error(y_test, y_pred_lr)
    r2_lr = r2_score(y_test, y_pred_lr)
    print(f"\nLinear Regression: MSE={mse_lr:.4f}, R²={r2_lr:.4f}")

    # Decision Tree
    dt = DecisionTreeRegressor(random_state=42).fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    mse_dt = mean_squared_error(y_test, y_pred_dt)
    r2_dt = r2_score(y_test, y_pred_dt)
    print(f"Decision Tree:      MSE={mse_dt:.4f}, R²={r2_dt:.4f}")

    metrics = {
        "linear_regression": {"mse": float(mse_lr), "r2": float(r2_lr)},
        "decision_tree": {"mse": float(mse_dt), "r2": float(r2_dt)},
        "linear_coefficients": {k: float(v) for k, v in zip(X.columns, lr.coef_)},
        "linear_intercept": float(lr.intercept_),
    }

    with open('metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print("\nSaved figures/ and metrics.json")
else:
    print("\nTarget or features missing — check your column names.")
