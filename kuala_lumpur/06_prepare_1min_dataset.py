"""
=============================================================
Dataset: 1-Minute GHI (Kuala Lumpur, Malaysia, 2025)
         + Open-Meteo Hourly Weather (interpolated to 1-min)
DOP: Forecasting of Renewable Resources Using AI
BITS Pilani, Hyderabad | EEE Dept | Prof. Arup Ratan
=============================================================
SOURCE 1: Local dataset — 1-min GHI, Kuala Lumpur 2025
  - 365 daily CSV files, 720 rows each (07:00-18:59)
  - Published in "Data in Brief" journal
SOURCE 2: Open-Meteo Historical Weather API (free, no key)
  - Hourly: Temperature, Humidity, Wind, Pressure, DNI, DHI
  - Interpolated to 1-minute resolution

APPROACH (Prof. Arup Ratan's meeting):
  - 8 features x 60 minutes = 480 data points per window
  - Rolling window forecast at minute-level resolution
  - Prove key features for this new dataset (as sir requested)
  - Train/test split: Jan-Sep 2025 train, Oct-Dec 2025 test

OUTPUT:
  kuala_lumpur_1min.csv  — merged 1-min dataset (all features)
=============================================================
"""

import os
import glob
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("=" * 65)
print("  DATA PREPARATION: 1-Minute Solar Dataset")
print("  Kuala Lumpur, Malaysia | 2025")
print("=" * 65)

# ── STEP 1: Load all GHI CSVs ─────────────────────────────────
print("\n[1/4] Loading 1-minute GHI data from 365 daily files...")

files = sorted(glob.glob(os.path.join(BASE_DIR, '2025', '**', '*.csv'), recursive=True))
print(f"  Found {len(files)} daily CSV files")

dfs = []
for f in files:
    try:
        df_day = pd.read_csv(f)
        dfs.append(df_day)
    except Exception as e:
        print(f"  [WARN] Could not read {f}: {e}")

df_ghi = pd.concat(dfs, ignore_index=True)
df_ghi.rename(columns={'Timestamps': 'DateTime', 'SolarIrradianceWm2': 'GHI'}, inplace=True)
df_ghi['DateTime'] = pd.to_datetime(df_ghi['DateTime'], dayfirst=True, errors='coerce')
df_ghi.dropna(subset=['DateTime'], inplace=True)
df_ghi.sort_values('DateTime', inplace=True)
df_ghi.reset_index(drop=True, inplace=True)
df_ghi['GHI'] = df_ghi['GHI'].clip(lower=0)

print(f"  Loaded: {len(df_ghi):,} rows")
print(f"  Date range: {df_ghi['DateTime'].min()} to {df_ghi['DateTime'].max()}")
print(f"  GHI range: {df_ghi['GHI'].min():.0f} to {df_ghi['GHI'].max():.0f} W/m2")

# ── STEP 2: Fetch Open-Meteo Hourly Weather for Kuala Lumpur ──
# Kuala Lumpur: lat=3.1390, lon=101.6869
KL_LAT = 3.1390
KL_LON = 101.6869

print("\n[2/4] Fetching hourly weather data from Open-Meteo...")
print("  (Temperature, Humidity, Wind, Pressure, DNI, DHI for 2025)")

WEATHER_CACHE = os.path.join(BASE_DIR, 'kl_hourly_weather.csv')

if os.path.exists(WEATHER_CACHE):
    print(f"  Loading cached weather data from {WEATHER_CACHE}...")
    df_weather = pd.read_csv(WEATHER_CACHE, parse_dates=['DateTime'])
else:
    # Download in chunks (Open-Meteo limit: ~1 year per request)
    url = (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={KL_LAT}&longitude={KL_LON}"
        f"&start_date=2025-01-01&end_date=2025-12-31"
        f"&hourly=shortwave_radiation,direct_normal_irradiance,diffuse_radiation,"
        f"temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,surface_pressure"
        f"&timezone=Asia%2FKuala_Lumpur&format=json"
    )
    print(f"  URL: {url[:90]}...")
    
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        hourly = data.get('hourly', {})
        
        if not hourly.get('time'):
            raise ValueError("Empty response from Open-Meteo")
        
        df_weather = pd.DataFrame({
            'DateTime':         pd.to_datetime(hourly['time']),
            'GHI_weather':      hourly.get('shortwave_radiation', [np.nan]*len(hourly['time'])),
            'DNI':              hourly.get('direct_normal_irradiance', [np.nan]*len(hourly['time'])),
            'DHI':              hourly.get('diffuse_radiation', [np.nan]*len(hourly['time'])),
            'Temperature':      hourly.get('temperature_2m', [np.nan]*len(hourly['time'])),
            'Relative Humidity':hourly.get('relative_humidity_2m', [np.nan]*len(hourly['time'])),
            'Wind Speed':       hourly.get('wind_speed_10m', [np.nan]*len(hourly['time'])),
            'Wind Direction':   hourly.get('wind_direction_10m', [np.nan]*len(hourly['time'])),
            'Pressure':         hourly.get('surface_pressure', [np.nan]*len(hourly['time'])),
        })
        df_weather.to_csv(WEATHER_CACHE, index=False)
        print(f"  Downloaded & cached: {len(df_weather):,} hourly rows")
        
    except Exception as e:
        print(f"  [WARN] Open-Meteo unavailable: {e}")
        print("  Falling back to synthetic weather generation...")
        
        # Synthetic fallback: Kuala Lumpur typical climate
        df_weather = None

# ── STEP 3: Interpolate Hourly Weather to 1-Minute ────────────
print("\n[3/4] Interpolating hourly weather to 1-minute resolution...")

# Create a full minute-level timestamp index matching GHI data
start_dt = df_ghi['DateTime'].min().replace(second=0, microsecond=0)
end_dt   = df_ghi['DateTime'].max().replace(second=0, microsecond=0)
minute_index = pd.date_range(start=start_dt, end=end_dt, freq='1min')

if df_weather is not None and len(df_weather) > 0:
    # Reindex weather to minute-level and interpolate
    df_weather = df_weather.set_index('DateTime').sort_index()
    df_weather_min = df_weather.reindex(
        df_weather.index.union(minute_index)
    ).interpolate(method='time').reindex(minute_index)
    df_weather_min.reset_index(inplace=True)
    df_weather_min.rename(columns={'index': 'DateTime'}, inplace=True)
    print(f"  Interpolated: {len(df_weather_min):,} minute-level weather rows")
    
    # Merge GHI (1-min actual) with weather (interpolated)
    df_ghi['DateTime'] = df_ghi['DateTime'].dt.round('min')
    df_merged = pd.merge(df_ghi, df_weather_min, on='DateTime', how='left')
    
    # Drop redundant GHI from weather (use actual 1-min GHI)
    if 'GHI_weather' in df_merged.columns:
        df_merged.drop(columns=['GHI_weather'], inplace=True)
        
else:
    # Synthetic weather for KL (tropical climate)
    print("  Generating synthetic tropical weather (KL baseline climate)...")
    df_merged = df_ghi.copy()
    
    # KL: near-equatorial, hot & humid year-round
    hour = df_merged['DateTime'].dt.hour + df_merged['DateTime'].dt.minute / 60
    doy  = df_merged['DateTime'].dt.dayofyear
    
    # Temperature: 26-34°C, peaks at 14:00
    df_merged['Temperature'] = 30 + 4 * np.sin((hour - 6) * np.pi / 12) + \
                                np.random.normal(0, 0.3, len(df_merged))
    df_merged['Temperature'] = df_merged['Temperature'].clip(22, 38)
    
    # Humidity: 70-95%, inversely correlated with temp
    df_merged['Relative Humidity'] = 90 - 10 * np.sin((hour - 6) * np.pi / 12) + \
                                     np.random.normal(0, 2, len(df_merged))
    df_merged['Relative Humidity'] = df_merged['Relative Humidity'].clip(55, 98)
    
    # Wind: 1-4 m/s
    df_merged['Wind Speed'] = 2 + np.random.exponential(0.8, len(df_merged))
    df_merged['Wind Speed'] = df_merged['Wind Speed'].clip(0, 10)
    
    # Wind direction: random
    df_merged['Wind Direction'] = np.random.uniform(0, 360, len(df_merged))
    
    # Pressure: ~1010 mbar (sea level)
    df_merged['Pressure'] = 1010 + np.random.normal(0, 2, len(df_merged))
    
    # DNI and DHI: derived from GHI (simplified)
    # For tropical: diffuse fraction ~0.4
    df_merged['DHI'] = (df_merged['GHI'] * 0.35).clip(0)
    df_merged['DNI'] = ((df_merged['GHI'] - df_merged['DHI']) * 1.05).clip(0)

# ── STEP 4: Add Time Features ─────────────────────────────────
print("\n[4/4] Engineering time and lag features...")

df_merged['Year']    = df_merged['DateTime'].dt.year
df_merged['Month']   = df_merged['DateTime'].dt.month
df_merged['Day']     = df_merged['DateTime'].dt.day
df_merged['Hour']    = df_merged['DateTime'].dt.hour
df_merged['Minute']  = df_merged['DateTime'].dt.minute
df_merged['DayOfYear'] = df_merged['DateTime'].dt.dayofyear

# Cyclical time encoding
df_merged['Hour_sin']   = np.sin(2 * np.pi * (df_merged['Hour'] * 60 + df_merged['Minute']) / 1440)
df_merged['Hour_cos']   = np.cos(2 * np.pi * (df_merged['Hour'] * 60 + df_merged['Minute']) / 1440)
df_merged['Month_sin']  = np.sin(2 * np.pi * df_merged['Month'] / 12)
df_merged['Month_cos']  = np.cos(2 * np.pi * df_merged['Month'] / 12)

# Lag features (1-minute, 5-minute, 10-minute, 30-minute lags)
df_merged.sort_values('DateTime', inplace=True)
df_merged.reset_index(drop=True, inplace=True)

df_merged['GHI_lag1']   = df_merged['GHI'].shift(1)   # 1 min ago
df_merged['GHI_lag5']   = df_merged['GHI'].shift(5)   # 5 min ago
df_merged['GHI_lag10']  = df_merged['GHI'].shift(10)  # 10 min ago
df_merged['GHI_lag30']  = df_merged['GHI'].shift(30)  # 30 min ago
df_merged['GHI_lag60']  = df_merged['GHI'].shift(60)  # 1 hour ago

# Rolling averages
df_merged['GHI_roll5']  = df_merged['GHI'].shift(1).rolling(5,  min_periods=1).mean()
df_merged['GHI_roll10'] = df_merged['GHI'].shift(1).rolling(10, min_periods=1).mean()
df_merged['GHI_roll30'] = df_merged['GHI'].shift(1).rolling(30, min_periods=1).mean()
df_merged['GHI_roll60'] = df_merged['GHI'].shift(1).rolling(60, min_periods=1).mean()

# Temperature lag
df_merged['Temp_lag5']  = df_merged['Temperature'].shift(5)

df_merged.dropna(inplace=True)
df_merged.reset_index(drop=True, inplace=True)

# ── Save ─────────────────────────────────────────────────────
OUT = os.path.join(BASE_DIR, 'kuala_lumpur_1min.csv')
df_merged.to_csv(OUT, index=False)

print("\n" + "=" * 65)
print("  DATASET SUMMARY: 1-Minute Kuala Lumpur Solar Data")
print("=" * 65)
print(f"  Total rows:    {len(df_merged):,}")
print(f"  Date range:    {df_merged['DateTime'].min()} to {df_merged['DateTime'].max()}")
print(f"  Columns ({df_merged.shape[1]}): {df_merged.columns.tolist()}")
print(f"  Daytime rows:  {(df_merged['GHI'] > 0).sum():,}")
print(f"  GHI range:     {df_merged['GHI'].min():.0f} – {df_merged['GHI'].max():.0f} W/m2")
print(f"  Temp range:    {df_merged['Temperature'].min():.1f} – {df_merged['Temperature'].max():.1f} deg C")
print(f"  Missing:       {df_merged.isnull().sum().sum()}")
print(f"\n  Saved: {OUT}  ({os.path.getsize(OUT)/1e6:.1f} MB)")
print("\n  Next steps:")
print("  1. Run 07_minute_eda.py        — EDA + feature importance")
print("  2. Run 08_minute_model.py      — Rolling window model")
print("=" * 65)
