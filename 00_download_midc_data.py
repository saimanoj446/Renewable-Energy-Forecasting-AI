"""
=============================================================
Dataset Acquisition: 1-Minute Solar Data (NREL MIDC SRRL BMS)
DOP: Forecasting of Renewable Resources Using AI
BITS Pilani, Hyderabad | EEE Dept | Prof. Arup Ratan
=============================================================
SOURCE: NREL Measurement & Instrumentation Data Center (MIDC)
STATION: Solar Radiation Research Laboratory (SRRL) — Golden, CO
RESOLUTION: 1-minute
COLUMNS: GHI, DNI, DHI, Temperature, Relative Humidity,
         Wind Speed, Wind Direction, Pressure
PERIOD: 2020-01-01 to 2022-12-31 (3 years, ~1.5M rows)

RELEVANCE (per Prof. Arup Ratan's meeting discussion):
  - 8 features × 60 minutes = 480 data points per window
  - Minute-level resolution for real-time forecasting concept
  - Proof that rolling window approach works at fine granularity
  - Sir said: "for proof of concept, any location data is fine"

HOW TO RUN:
  python 00_download_midc_data.py
  (will take ~5-10 minutes, downloads ~150 daily files)

OUTPUT:
  midc_1min_raw/     — raw daily CSVs
  midc_1min_data.csv — merged, cleaned minute-level dataset
=============================================================
"""

import os
import time
import requests
import pandas as pd
import numpy as np
from datetime import date, timedelta
from io import StringIO

# ── Config ────────────────────────────────────────────────────
SITE      = 'BMS'                    # SRRL Baseline Measurement System, Golden CO
START_DT  = date(2020, 1, 1)
END_DT    = date(2022, 12, 31)
RAW_DIR   = 'midc_1min_raw'
OUT_CSV   = 'midc_1min_data.csv'
BASE_URL  = 'https://midcdmz.nrel.gov/apps/daily.pl'
DELAY_SEC = 0.5                      # polite delay between requests

os.makedirs(RAW_DIR, exist_ok=True)

print("=" * 65)
print("  NREL MIDC DATA DOWNLOADER")
print("  Station: SRRL BMS — Golden, Colorado")
print(f"  Period:  {START_DT} → {END_DT}")
print(f"  Resolution: 1-minute")
print("=" * 65)

# ── Column name mapper ────────────────────────────────────────
# MIDC column names can vary; we map common variants to standard names
COL_MAP = {
    # GHI variants
    'Global CMP22 (vent/heat) [W/m^2]':   'GHI',
    'Global PSP [W/m^2]':                  'GHI',
    'Global Horiz (platform) [W/m^2]':     'GHI',
    'Global Horiz [W/m^2]':                'GHI',
    'Global PSP2 [W/m^2]':                 'GHI',
    # DNI variants
    'Direct NIP [W/m^2]':                  'DNI',
    'Direct NIP2 [W/m^2]':                 'DNI',
    'Direct (shadow band) [W/m^2]':        'DNI',
    # DHI variants
    'Diffuse 8-48 [W/m^2]':               'DHI',
    'Diffuse CM22 [W/m^2]':               'DHI',
    'Diffuse [W/m^2]':                     'DHI',
    # Temperature
    'Tower Dry Bulb Temp [deg C]':         'Temperature',
    'Dry Bulb Temp [deg C]':               'Temperature',
    'Ambient Temperature [deg C]':         'Temperature',
    # Relative Humidity
    'Tower Relative Humidity [%]':         'Relative Humidity',
    'Relative Humidity [%]':               'Relative Humidity',
    # Wind Speed
    'Avg Wind Speed @ 10m [m/s]':          'Wind Speed',
    'Wind Speed [m/s]':                    'Wind Speed',
    'Avg Wind Speed [m/s]':                'Wind Speed',
    # Wind Direction
    'Avg Wind Direction @ 10m [deg from N]': 'Wind Direction',
    'Wind Direction [deg from N]':           'Wind Direction',
    # Pressure
    'Station Pressure [mBar]':             'Pressure',
    'Avg Barometric Pressure [mBar]':      'Pressure',
    'Barometric Pressure [mBar]':          'Pressure',
}
REQUIRED_COLS = ['GHI', 'DNI', 'DHI', 'Temperature',
                 'Relative Humidity', 'Wind Speed', 'Wind Direction', 'Pressure']

# ── Download daily files ──────────────────────────────────────
current = START_DT
total_days = (END_DT - START_DT).days + 1
downloaded = 0
skipped    = 0
failed     = 0

print(f"\nDownloading {total_days} daily files...")
all_dfs = []

while current <= END_DT:
    yr, mo, dy = current.year, current.month, current.day
    fname = os.path.join(RAW_DIR, f'midc_{yr}{mo:02d}{dy:02d}.csv')
    
    if os.path.exists(fname):
        # Already downloaded — load from cache
        try:
            raw = open(fname, 'r').read()
        except:
            raw = ''
        skipped += 1
    else:
        url = (f"{BASE_URL}?site={SITE}&start=20000101"
               f"&yr={yr}&mo={mo}&dy={dy}")
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                raw = resp.text
                with open(fname, 'w') as f:
                    f.write(raw)
                downloaded += 1
            else:
                print(f"  [WARN] HTTP {resp.status_code} for {current}")
                failed += 1
                raw = ''
            time.sleep(DELAY_SEC)
        except Exception as e:
            print(f"  [ERROR] {current}: {e}")
            failed += 1
            raw = ''
    
    # Parse the CSV
    if raw:
        try:
            # MIDC CSVs have a header line then data
            lines = [l for l in raw.strip().split('\n') if l.strip()]
            if len(lines) >= 2:
                df_day = pd.read_csv(StringIO('\n'.join(lines)))
                
                # Rename columns
                df_day.rename(columns=COL_MAP, inplace=True)
                df_day.rename(columns={c: c.strip() for c in df_day.columns}, inplace=True)
                
                # Parse datetime (MIDC format: Date, MST, or Time column)
                datetime_col = None
                for dc in ['Date', 'DATE']:
                    if dc in df_day.columns:
                        datetime_col = dc
                        break
                time_col = None
                for tc in ['MST', 'Time', 'TIME']:
                    if tc in df_day.columns:
                        time_col = tc
                        break
                
                if datetime_col and time_col:
                    df_day['DateTime'] = pd.to_datetime(
                        df_day[datetime_col].astype(str) + ' ' + df_day[time_col].astype(str),
                        errors='coerce'
                    )
                elif 'DateTime' in df_day.columns:
                    df_day['DateTime'] = pd.to_datetime(df_day['DateTime'], errors='coerce')
                else:
                    df_day['DateTime'] = pd.to_datetime(
                        f"{yr}-{mo:02d}-{dy:02d}", errors='coerce'
                    )
                
                df_day.dropna(subset=['DateTime'], inplace=True)
                
                # Keep only standard columns that exist
                keep = ['DateTime'] + [c for c in REQUIRED_COLS if c in df_day.columns]
                df_day = df_day[keep].copy()
                
                # Remove rows where all sensor values are NaN or negative
                sensor_cols = [c for c in REQUIRED_COLS if c in df_day.columns]
                if sensor_cols:
                    all_dfs.append(df_day)
        except Exception as e:
            if failed < 5:
                print(f"  [PARSE ERROR] {current}: {e}")
    
    current += timedelta(days=1)
    
    # Progress bar every 30 days
    total_done = downloaded + skipped + failed
    if total_done % 30 == 0:
        pct = total_done / total_days * 100
        print(f"  Progress: {total_done}/{total_days} ({pct:.0f}%) | "
              f"downloaded={downloaded} cached={skipped} failed={failed}")

print(f"\nDownload complete.")
print(f"  Downloaded: {downloaded}, Cached: {skipped}, Failed: {failed}")

if not all_dfs:
    print("\n[ERROR] No data frames could be parsed. Possible issues:")
    print("  1. Network connectivity problem (check internet)")
    print("  2. MIDC site may be temporarily unavailable")
    print("  3. Column name format may have changed")
    print("\nAlternative: Use the NSRDB 5-min data download script instead.")
    print("See: 00b_download_nsrdb.py")
    exit(1)

# ── Merge all days ────────────────────────────────────────────
print("\nMerging all daily files...")
df = pd.concat(all_dfs, ignore_index=True)
df.sort_values('DateTime', inplace=True)
df.reset_index(drop=True, inplace=True)

print(f"Raw merged: {len(df):,} rows × {df.shape[1]} columns")
print(f"Date range: {df['DateTime'].min()} to {df['DateTime'].max()}")
print(f"Available columns: {df.columns.tolist()}")

# ── Add time features ────────────────────────────────────────
df['Year']    = df['DateTime'].dt.year
df['Month']   = df['DateTime'].dt.month
df['Day']     = df['DateTime'].dt.day
df['Hour']    = df['DateTime'].dt.hour
df['Minute']  = df['DateTime'].dt.minute
df['DayOfYear'] = df['DateTime'].dt.dayofyear

# Clean: clip negatives for irradiance
for col in ['GHI', 'DNI', 'DHI']:
    if col in df.columns:
        df[col] = df[col].clip(lower=0)

# Drop rows with all-sensor NaN
sensor_check = [c for c in ['GHI', 'Temperature'] if c in df.columns]
df.dropna(subset=sensor_check, inplace=True)

# ── Summary ──────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  DATASET SUMMARY (1-minute NREL MIDC data)")
print("=" * 65)
print(f"  Total rows:    {len(df):,}")
print(f"  Columns:       {df.columns.tolist()}")
print(f"  Date range:    {df['DateTime'].min()} to {df['DateTime'].max()}")
print(f"  Missing values:\n{df[sensor_check].isnull().sum().to_string()}")
if 'GHI' in df.columns:
    daytime = df[df['GHI'] > 0]
    print(f"  Daytime rows:  {len(daytime):,} / {len(df):,}")
    print(f"  GHI range:     {df['GHI'].min():.0f} – {df['GHI'].max():.0f} W/m²")
if 'Temperature' in df.columns:
    print(f"  Temp range:    {df['Temperature'].min():.1f} – {df['Temperature'].max():.1f} °C")

# ── Save ─────────────────────────────────────────────────────
df.to_csv(OUT_CSV, index=False)
print(f"\n  Saved to: {OUT_CSV}  ({os.path.getsize(OUT_CSV)/1e6:.1f} MB)")
print("\n  Next step: run 06_minute_level_eda.py")
print("=" * 65)
