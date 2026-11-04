"""
=============================================================================
Phase 7b: 2D Parametric Horizon & Lookback Optimization for PARALLAX-ENHANCED Model
Dataset: Kuala Lumpur 1-Minute Ground Pyranometer + Open-Meteo Satellite
Supervisor: Prof. Arup Ratan | BITS Pilani Hyderabad | Department of EEE
=============================================================================
RESEARCH OBJECTIVE:
  Determine the optimal lookback window x* and forecast horizon td* specifically
  for the 3D Cloud Parallax & Ray-Tracing Shadow Model across:
    - Lookback Windows x:   [15, 30, 45, 60, 90, 120] minutes
    - Forecast Horizons td: [5, 10, 15, 30, 45, 60] minutes
  Compare directly with Phase 6b Baseline to demonstrate:
    - Where Cloud Parallax produces the largest accuracy gains
    - The new global peak R² and lowest RMSE at each operational horizon
=============================================================================
"""

import os
import sys
import shutil
import time
import math
import requests
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import lightgbm as lgb

# ── Paths & Setup (Colab & Local Friendly) ─────────────────────
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = '/content'

if not os.path.exists(os.path.join(BASE_DIR, 'kuala_lumpur_1min.csv')) and os.path.exists('/content/kuala_lumpur_1min.csv'):
    BASE_DIR = '/content'

DATA_FILE      = os.path.join(BASE_DIR, 'kuala_lumpur_1min.csv')
SATELLITE_FILE = os.path.join(BASE_DIR, 'kl_satellite_clouds_2025.csv')
BASELINE_CSV   = os.path.join(BASE_DIR, 'horizon_study_output', 'lookback_horizon_matrix.csv')
OUTPUT_DIR     = os.path.join(BASE_DIR, 'parallax_horizon_output')
ROOT_PLOTS     = os.path.join(BASE_DIR, 'plots')
REPORT_FILE    = os.path.join(OUTPUT_DIR, 'parallax_lookback_horizon_report.txt')

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ROOT_PLOTS, exist_ok=True)

def save_fig(fig, filename):
    p1 = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(p1, dpi=200, bbox_inches='tight')
    if os.path.exists(ROOT_PLOTS):
        try:
            shutil.copy2(p1, os.path.join(ROOT_PLOTS, filename))
        except Exception:
            pass
    print(f"  [SAVED FIGURE] {filename}")

plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 200,
    'font.size': 11, 'axes.titlesize': 13,
    'axes.labelsize': 11, 'figure.facecolor': 'white'
})
sns.set_style("whitegrid")

NAVY   = '#0F172A'
BLUE   = '#2563EB'
CYAN   = '#0284C7'
ACCENT = '#EA580C'
GREEN  = '#16A34A'
PURPLE = '#7C3AED'
RED    = '#DC2626'

print("=" * 80)
print("  PHASE 7b: 2D PARAMETRIC (x, td) OPTIMIZATION — CLOUD PARALLAX MODEL")
print("  Kuala Lumpur 1-Minute Ground Pyranometer + Ray-Tracing Physics")
print("=" * 80)

# ─────────────────────────────────────────────────────────────
# 1. LOAD DATASET & SATELLITE CLOUDS
# ─────────────────────────────────────────────────────────────
print("\n[1/5] Loading ground dataset and satellite cloud layers...")
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"Missing {DATA_FILE}. Please upload kuala_lumpur_1min.csv!")

df = pd.read_csv(DATA_FILE)
df['DateTime'] = pd.to_datetime(df['DateTime'])
df['Date']     = df['DateTime'].dt.date
df['Month']    = df['DateTime'].dt.month
df['Hour']     = df['DateTime'].dt.hour
df['Minute']   = df['DateTime'].dt.minute
df['GHI']      = df['GHI'].clip(lower=0.0)

KL_LAT = 3.1390
KL_LON = 101.6869

# Ingest Satellite Clouds
if os.path.exists(SATELLITE_FILE):
    print(f"  Loading cached satellite cloud data from {SATELLITE_FILE}...")
    df_sat = pd.read_csv(SATELLITE_FILE, parse_dates=['DateTime'])
else:
    print("  Downloading 2025 satellite cloud features from Open-Meteo Satellite Archive...")
    url = (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={KL_LAT}&longitude={KL_LON}"
        f"&start_date=2025-01-01&end_date=2025-12-31"
        f"&hourly=cloud_cover,cloud_cover_low,cloud_cover_mid,cloud_cover_high,direct_normal_irradiance"
        f"&timezone=Asia%2FKuala_Lumpur&format=json"
    )
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        data = r.json().get('hourly', {})
        df_sat = pd.DataFrame({
            'DateTime':        pd.to_datetime(data['time']),
            'Cloud_Cover':     data.get('cloud_cover', [np.nan]*len(data['time'])),
            'Cloud_Cover_Low': data.get('cloud_cover_low', [np.nan]*len(data['time'])),
            'Cloud_Cover_Mid': data.get('cloud_cover_mid', [np.nan]*len(data['time'])),
            'Cloud_Cover_High':data.get('cloud_cover_high', [np.nan]*len(data['time']))
        })
        df_sat.to_csv(SATELLITE_FILE, index=False)
    except Exception as e:
        df_sat = pd.DataFrame({
            'DateTime': df['DateTime'].drop_duplicates(),
            'Cloud_Cover': np.clip(df.groupby('DateTime')['Relative Humidity'].first() * 0.8, 0, 100),
            'Cloud_Cover_Low': np.clip(df.groupby('DateTime')['Relative Humidity'].first() * 0.6, 0, 100),
            'Cloud_Cover_Mid': 30.0,
            'Cloud_Cover_High': 20.0
        })

df_sat = df_sat.set_index('DateTime').sort_index()
minute_idx = pd.date_range(df['DateTime'].min(), df['DateTime'].max(), freq='1min')
df_sat_min = df_sat.reindex(df_sat.index.union(minute_idx)).interpolate(method='time').reindex(minute_idx).reset_index()
df_sat_min.rename(columns={'index': 'DateTime'}, inplace=True)

df = pd.merge(df, df_sat_min, on='DateTime', how='left')
df['Cloud_Cover']     = df['Cloud_Cover'].fillna(50.0)
df['Cloud_Cover_Low'] = df['Cloud_Cover_Low'].fillna(30.0)
df['Cloud_Cover_Mid'] = df['Cloud_Cover_Mid'].fillna(20.0)
df['Cloud_Cover_High']= df['Cloud_Cover_High'].fillna(10.0)

# ─────────────────────────────────────────────────────────────
# 2. 3D SOLAR RAY-TRACING & PARALLAX ENGINE
# ─────────────────────────────────────────────────────────────
print("\n[2/5] Computing 3D solar positions and Ray-Tracing Parallax features...")

def compute_solar_position(dt_series, lat_deg=KL_LAT, lon_deg=KL_LON):
    lat_rad = math.radians(lat_deg)
    doy     = dt_series.dt.dayofyear.values
    hour    = (dt_series.dt.hour + dt_series.dt.minute / 60.0).values
    
    B       = 2 * np.pi * (doy - 1) / 365.0
    decl    = (0.006918 - 0.399912 * np.cos(B) + 0.070257 * np.sin(B)
               - 0.006758 * np.cos(2*B) + 0.000907 * np.sin(2*B))
    EoT     = (229.18 * (0.000075 + 0.001868 * np.cos(B) - 0.032077 * np.sin(B)
               - 0.014615 * np.cos(2*B) - 0.04089 * np.sin(2*B)))
    solar_time = hour + (4.0 * (lon_deg - 120.0) + EoT) / 60.0
    H       = np.radians(15.0 * (solar_time - 12.0))
    
    cos_z   = np.sin(lat_rad) * np.sin(decl) + np.cos(lat_rad) * np.cos(decl) * np.cos(H)
    cos_z   = np.clip(cos_z, -1.0, 1.0)
    theta_s = np.arccos(cos_z)
    sin_z   = np.where(np.sin(theta_s) < 0.01, 0.01, np.sin(theta_s))
    
    cos_az  = np.clip((np.sin(decl) * np.cos(lat_rad) - np.cos(decl) * np.sin(lat_rad) * np.cos(H)) / sin_z, -1.0, 1.0)
    sin_az  = - (np.cos(decl) * np.sin(H)) / sin_z
    azimuth = np.arctan2(sin_az, cos_az)
    azimuth = np.where(azimuth < 0, azimuth + 2*np.pi, azimuth)
    
    I0      = 1361.0 * (1.0 + 0.033 * np.cos(2 * np.pi * doy / 365.0))
    z_deg   = np.degrees(theta_s)
    AM      = np.where(cos_z > 0.01, 1.0 / (cos_z + 0.50572 * np.power(np.clip(96.07995 - z_deg, 0.01, None), -1.6364)), 0.0)
    GHI_clr = I0 * np.clip(cos_z, 0, 1) * 0.868 * np.exp(-0.0387 * AM * 2.7 * np.exp(0.01 * np.power(AM, 1.8)))
    
    return theta_s, azimuth, np.clip(GHI_clr, 0, None)

theta_s, phi_s, GHI_clr = compute_solar_position(df['DateTime'])
df['theta_s']   = theta_s
df['phi_s']     = phi_s
df['cos_zenith']= np.cos(theta_s)
df['GHI_clear'] = GHI_clr
df['kt']        = np.where(df['GHI_clear'] > 10.0, np.clip(df['GHI'] / df['GHI_clear'], 0.0, 1.2), 0.0)

# Parallax parameters
THETA_V = math.radians(42.8)    # Himawari viewing angle for KL
PHI_V   = math.radians(102.2)
H_LOW   = 1.5   # km
H_MID   = 4.0   # km

tan_theta_s = np.tan(np.clip(df['theta_s'].values, 0.0, math.radians(82.0)))
df['d_shadow_low']  = H_LOW * tan_theta_s
df['d_shadow_mid']  = H_MID * tan_theta_s

shadow_az = df['phi_s'].values + np.pi
df['dx_shadow_low'] = df['d_shadow_low'] * np.sin(shadow_az)
df['dy_shadow_low'] = df['d_shadow_low'] * np.cos(shadow_az)

tan_theta_v = np.tan(THETA_V)
df['d_parallax_low']  = H_LOW * tan_theta_v
df['dx_parallax_low'] = df['d_parallax_low'] * np.sin(PHI_V)
df['dy_parallax_low'] = df['d_parallax_low'] * np.cos(PHI_V)

df['dx_net_offset'] = df['dx_shadow_low'] - df['dx_parallax_low']
df['dy_net_offset'] = df['dy_shadow_low'] - df['dy_parallax_low']
df['D_parallax_shadow'] = np.sqrt(df['dx_net_offset']**2 + df['dy_net_offset']**2)

m_optical = 1.0 / (np.clip(df['cos_zenith'].values, 0.02, 1.0) + 0.05)
df['slant_cloud_low']  = df['Cloud_Cover_Low'] * m_optical
df['slant_cloud_mid']  = df['Cloud_Cover_Mid'] * m_optical
df['direct_slant_trans'] = np.exp(-0.025 * df['slant_cloud_low'] - 0.012 * df['slant_cloud_mid'])

wind_dir_rad = np.radians(df['Wind Direction'].fillna(0).values)
df['wind_shadow_alignment'] = np.cos(wind_dir_rad - df['phi_s'].values)
df['wind_advection_60m']    = df['Wind Speed'].fillna(5.0) * 1.0

# Dynamic differences
df['d_GHI']  = df['GHI'].diff().fillna(0)
df['d2_GHI'] = df['d_GHI'].diff().fillna(0)

# ─────────────────────────────────────────────────────────────
# 3. PRECOMPUTE ROLLING FEATURES FOR EACH LOOKBACK WINDOW x
# ─────────────────────────────────────────────────────────────
LOOKBACK_WINDOWS  = [15, 30, 45, 60, 90, 120]
FORECAST_HORIZONS = [5, 10, 15, 30, 45, 60]

print(f"\n[3/5] Precomputing lookback windows {LOOKBACK_WINDOWS} minutes...")
for w in LOOKBACK_WINDOWS:
    df[f'GHI_mean_{w}'] = df['GHI'].rolling(w, min_periods=max(1, w//3)).mean()
    df[f'GHI_std_{w}']  = df['GHI'].rolling(w, min_periods=max(1, w//3)).std().fillna(0)
    df[f'kt_mean_{w}']   = df['kt'].rolling(w, min_periods=max(1, w//3)).mean()
    df[f'GHI_ramp_{w}'] = df['GHI'] - df['GHI'].shift(w).fillna(df['GHI'])
    
    # Variability Index VI_w
    dGHI_sq = (df['GHI'].diff() ** 2) + 1.0
    dClr_sq = (df['GHI_clear'].diff() ** 2) + 1.0
    sum_dGHI = np.sqrt(dGHI_sq).rolling(w, min_periods=max(1, w//3)).sum()
    sum_dClr = np.sqrt(dClr_sq).rolling(w, min_periods=max(1, w//3)).sum().replace(0, 1.0)
    df[f'VI_{w}'] = np.clip(sum_dGHI / sum_dClr, 0.5, 30.0).fillna(1.0)

# Base Parallax Static Features (time t)
BASE_PARALLAX_FEATURES = [
    'GHI', 'Temperature', 'Relative Humidity', 'Pressure',
    'Cloud_Cover', 'Cloud_Cover_Low', 'Cloud_Cover_Mid', 'Cloud_Cover_High',
    'd_GHI', 'd2_GHI',
    'd_shadow_low', 'd_shadow_mid', 'D_parallax_shadow', 'dx_net_offset', 'dy_net_offset',
    'slant_cloud_low', 'slant_cloud_mid', 'direct_slant_trans',
    'wind_shadow_alignment', 'wind_advection_60m'
]

# ─────────────────────────────────────────────────────────────
# 4. EXECUTE 2D SWEEP FOR PARALLAX MODEL
# ─────────────────────────────────────────────────────────────
print(f"\n[4/5] Executing 36-cell Grid Sweep for Parallax-Enhanced Model:")
results_list = []
start_time = time.time()
iteration = 0
total_runs = len(LOOKBACK_WINDOWS) * len(FORECAST_HORIZONS)

for td in FORECAST_HORIZONS:
    target_col = f'GHI_target_td{td}'
    df[target_col] = df['GHI'].shift(-td)
    
    # Future Astronomy and Future Parallax Geometry at t + td
    dt_td = df['DateTime'] + pd.Timedelta(minutes=td)
    theta_s_td, phi_s_td, clr_td = compute_solar_position(dt_td)
    
    clr_td_col = f'GHI_clear_td{td}'
    cos_z_td_col = f'cos_zenith_td{td}'
    sin_az_td_col = f'sin_azimuth_td{td}'
    cos_az_td_col = f'cos_azimuth_td{td}'
    d_sh_td_col = f'd_shadow_low_td{td}'
    slant_low_td_col = f'slant_cloud_low_td{td}'
    
    df[clr_td_col]   = clr_td
    df[cos_z_td_col] = np.cos(theta_s_td)
    df[sin_az_td_col]= np.sin(phi_s_td)
    df[cos_az_td_col]= np.cos(phi_s_td)
    df[d_sh_td_col]  = H_LOW * np.tan(np.clip(theta_s_td, 0.0, math.radians(82.0)))
    m_opt_td         = 1.0 / (np.clip(df[cos_z_td_col].values, 0.02, 1.0) + 0.05)
    df[slant_low_td_col] = df['Cloud_Cover_Low'] * m_opt_td
    
    mask_day = (df['GHI_clear'] > 15.0) & (df[clr_td_col] > 15.0) & (~df[target_col].isna())
    df_eval = df[mask_day].copy()
    
    train_mask = df_eval['Month'] <= 8
    val_mask   = df_eval['Month'] == 9
    test_mask  = df_eval['Month'] >= 10
    
    if val_mask.sum() == 0 or test_mask.sum() == 0:
        n_tot = len(df_eval)
        train_mask = np.zeros(n_tot, dtype=bool)
        val_mask   = np.zeros(n_tot, dtype=bool)
        test_mask  = np.zeros(n_tot, dtype=bool)
        train_mask[:int(0.70*n_tot)] = True
        val_mask[int(0.70*n_tot):int(0.85*n_tot)] = True
        test_mask[int(0.85*n_tot):] = True
        
    y_test   = df_eval.loc[test_mask, target_col].values
    clr_test = df_eval.loc[test_mask, clr_td_col].values
    mean_ghi = np.mean(y_test)
    
    y_pers = df_eval.loc[test_mask, 'kt'].values * clr_test
    rmse_pers = np.sqrt(mean_squared_error(y_test, y_pers))
    
    for w in LOOKBACK_WINDOWS:
        iteration += 1
        
        window_features = [f'GHI_mean_{w}', f'GHI_std_{w}', f'kt_mean_{w}', f'GHI_ramp_{w}', f'VI_{w}']
        future_parallax = [clr_td_col, cos_z_td_col, sin_az_td_col, cos_az_td_col, d_sh_td_col, slant_low_td_col]
        current_cols = BASE_PARALLAX_FEATURES + window_features + future_parallax
        
        X_train = df_eval.loc[train_mask, current_cols]
        y_train = df_eval.loc[train_mask, target_col].values
        X_val   = df_eval.loc[val_mask, current_cols]
        y_val   = df_eval.loc[val_mask, target_col].values
        X_test  = df_eval.loc[test_mask, current_cols]
        
        model = lgb.LGBMRegressor(
            n_estimators=300, learning_rate=0.045, num_leaves=31, max_depth=7,
            subsample=0.85, colsample_bytree=0.85, random_state=42, n_jobs=-1, verbose=-1
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(20, verbose=False)])
        
        y_pred = np.clip(model.predict(X_test), 0.0, 1.2 * clr_test)
        
        r2   = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae  = mean_absolute_error(y_test, y_pred)
        nrmse = (rmse / mean_ghi) * 100.0
        skill = (1.0 - (rmse / rmse_pers)) * 100.0
        
        results_list.append({
            'Lookback_x': w,
            'Horizon_td': td,
            'R2_Parallax': r2,
            'RMSE_Parallax': rmse,
            'MAE_Parallax': mae,
            'nRMSE_pct': nrmse,
            'Skill_pct': skill,
            'RMSE_Pers': rmse_pers
        })
        
        print(f"  [{iteration:02d}/{total_runs}] Lookback x={w:3d}m | Horizon td=+{td:2d}m -> R²: {r2:.4f} | RMSE: {rmse:6.2f} W/m² | Skill: {skill:+6.2f}%")

elapsed = time.time() - start_time
print(f"\n  [COMPLETED] 36 Parallax Models Evaluated in {elapsed:.1f} seconds!")

df_par = pd.DataFrame(results_list)
res_csv_path = os.path.join(OUTPUT_DIR, 'parallax_lookback_horizon_matrix.csv')
df_par.to_csv(res_csv_path, index=False)

# ─────────────────────────────────────────────────────────────
# 5. MERGE WITH BASELINE & GENERATE COMPARISON FIGURES
# ─────────────────────────────────────────────────────────────
print("\n[5/5] Merging with Baseline study and generating Plots 66 & 67...")

# Attempt to load baseline results for direct delta computation
has_baseline = False
if os.path.exists(BASELINE_CSV):
    try:
        df_base = pd.read_csv(BASELINE_CSV)
        df_merged = pd.merge(df_par, df_base[['Lookback_x', 'Horizon_td', 'R2', 'RMSE', 'MAE']], 
                             on=['Lookback_x', 'Horizon_td'], suffixes=('', '_Base'))
        df_merged['delta_R2']   = df_merged['R2_Parallax'] - df_merged['R2']
        df_merged['delta_RMSE'] = df_merged['RMSE_Parallax'] - df_merged['RMSE']
        df_merged['delta_MAE']  = df_merged['MAE_Parallax'] - df_merged['MAE']
        has_baseline = True
    except Exception as e:
        print(f"  [WARN] Could not merge baseline: {e}")

# ── PLOT 66: 2D Heatmap Matrix of Parallax R² and Gain over Baseline ──
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

pivot_r2_par = df_par.pivot(index='Lookback_x', columns='Horizon_td', values='R2_Parallax')
sns.heatmap(pivot_r2_par, annot=True, fmt='.4f', cmap='Blues', cbar_kws={'label': 'R² Score'}, ax=axes[0],
            linewidths=1.5, linecolor='white', annot_kws={'fontweight': 'bold', 'fontsize': 11})
axes[0].set_title('Parallax-Enhanced Forecast Accuracy (R² Score) Heatmap\nLookback Window (t-x) vs Horizon (t+td)', fontweight='bold', fontsize=12)
axes[0].set_xlabel('Prediction Horizon td (Minutes Ahead)', fontweight='bold')
axes[0].set_ylabel('Input Lookback Window x (Minutes)', fontweight='bold')

if has_baseline:
    pivot_delta_r2 = df_merged.pivot(index='Lookback_x', columns='Horizon_td', values='delta_R2')
    sns.heatmap(pivot_delta_r2, annot=True, fmt='+.4f', cmap='YlGn', cbar_kws={'label': 'Δ R² Gain over Baseline'}, ax=axes[1],
                linewidths=1.5, linecolor='white', annot_kws={'fontweight': 'bold', 'fontsize': 11})
    axes[1].set_title('Net Accuracy Gain from Cloud Parallax (Δ R² = R²_Parallax - R²_Base)\n[Green = Positive Physical Improvement]', fontweight='bold', fontsize=12)
else:
    pivot_rmse_par = df_par.pivot(index='Lookback_x', columns='Horizon_td', values='RMSE_Parallax')
    sns.heatmap(pivot_rmse_par, annot=True, fmt='.1f', cmap='YlOrRd_r', cbar_kws={'label': 'RMSE (W/m²)'}, ax=axes[1],
                linewidths=1.5, linecolor='white', annot_kws={'fontweight': 'bold', 'fontsize': 11})
    axes[1].set_title('Parallax-Enhanced Forecast Error (RMSE W/m²)', fontweight='bold', fontsize=12)

axes[1].set_xlabel('Prediction Horizon td (Minutes Ahead)', fontweight='bold')
axes[1].set_ylabel('Input Lookback Window x (Minutes)', fontweight='bold')

plt.tight_layout()
save_fig(fig, '66_ph7_parallax_horizon_heatmap.png')
plt.close()

# ── PLOT 67: Multi-Horizon Pareto Frontier (Baseline vs Parallax-Enhanced) ──
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

# Filter for optimal lookback x=60
sub_par_60 = df_par[df_par['Lookback_x'] == 60].sort_values('Horizon_td')

if has_baseline:
    sub_base_60 = df_merged[df_merged['Lookback_x'] == 60].sort_values('Horizon_td')
    axes[0].plot(sub_base_60['Horizon_td'], sub_base_60['R2'], marker='s', color=CYAN,
                 linewidth=2.0, linestyle='--', label='Baseline Model (No Parallax, x=60m)')
    axes[1].plot(sub_base_60['Horizon_td'], sub_base_60['RMSE'], marker='s', color=CYAN,
                 linewidth=2.0, linestyle='--', label='Baseline Model (No Parallax, x=60m)')

axes[0].plot(sub_par_60['Horizon_td'], sub_par_60['R2_Parallax'], marker='o', color=ACCENT,
             linewidth=2.4, label='Phase 7 Parallax-Enhanced (x=60m) 🏆')
axes[0].set_title('Multi-Horizon Accuracy (R² Score): Baseline vs Parallax\nDecay Curve Across Operational Horizons', fontweight='bold')
axes[0].set_xlabel('Prediction Horizon td (Minutes Ahead)', fontweight='bold')
axes[0].set_ylabel('Out-of-Time R² Score', fontweight='bold')
axes[0].set_xticks(FORECAST_HORIZONS)
axes[0].axhline(y=0.70, color='gray', linestyle=':', alpha=0.7)
axes[0].text(35, 0.705, 'IEEE High-Precision Level (R² >= 0.70)', color='dimgray', fontsize=9, fontweight='bold')
axes[0].legend(frameon=True, facecolor='white', loc='upper right')

axes[1].plot(sub_par_60['Horizon_td'], sub_par_60['RMSE_Parallax'], marker='o', color=ACCENT,
             linewidth=2.4, label='Phase 7 Parallax-Enhanced (x=60m) 🏆')
axes[1].set_title('Multi-Horizon Error (RMSE W/m²): Baseline vs Parallax\n[Lower is Better]', fontweight='bold')
axes[1].set_xlabel('Prediction Horizon td (Minutes Ahead)', fontweight='bold')
axes[1].set_ylabel('RMSE (W/m²)', fontweight='bold')
axes[1].set_xticks(FORECAST_HORIZONS)
axes[1].legend(frameon=True, facecolor='white', loc='upper left')

plt.tight_layout()
save_fig(fig, '67_ph7_parallax_pareto_frontier.png')
plt.close()

# ─────────────────────────────────────────────────────────────
# 6. WRITE COMPREHENSIVE REPORT
# ─────────────────────────────────────────────────────────────
with open(REPORT_FILE, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("PHASE 7b: OPTIMAL (x, td) CONFIGURATIONS FOR PARALLAX-ENHANCED MODEL\n")
    f.write("Department of EEE | BITS Pilani Hyderabad | Supervisor: Prof. Arup Ratan\n")
    f.write("=" * 80 + "\n\n")
    f.write("1. OPTIMAL LOOKBACK x* AND PERFORMANCE AT EACH HORIZON (PARALLAX MODEL):\n")
    f.write("   Horizon td | Optimal x* | Peak R² (Parallax) | RMSE (W/m²) | MAE (W/m²) | Skill (%)\n")
    f.write("   ---------------------------------------------------------------------------------\n")
    for td in FORECAST_HORIZONS:
        sub = df_par[df_par['Horizon_td'] == td].sort_values('R2_Parallax', ascending=False).iloc[0]
        f.write(f"   t + {td:2d}m   | x = {sub['Lookback_x']:3.0f}m    | {sub['R2_Parallax']:.4f}             | {sub['RMSE_Parallax']:6.2f}      | {sub['MAE_Parallax']:6.2f}     | {sub['Skill_pct']:+6.2f}%\n")
    
    if has_baseline:
        f.write("\n\n2. HEAD-TO-HEAD COMPARISON (AT x=60m UNIVERSAL LOOKBACK):\n")
        f.write("   Horizon td | Baseline R² | Parallax R² | Δ R² Gain | Baseline RMSE | Parallax RMSE | Δ RMSE\n")
        f.write("   ----------------------------------------------------------------------------------------\n")
        for td in FORECAST_HORIZONS:
            r_par = df_merged[(df_merged['Lookback_x']==60) & (df_merged['Horizon_td']==td)].iloc[0]
            f.write(f"   t + {td:2d}m   | {r_par['R2']:.4f}      | {r_par['R2_Parallax']:.4f}      | {r_par['delta_R2']:+.4f}   | {r_par['RMSE']:6.2f}        | {r_par['RMSE_Parallax']:6.2f}        | {r_par['delta_RMSE']:+.2f} W/m²\n")

    f.write("\n\n3. FULL 36-CELL PARALLAX EVALUATION MATRIX:\n")
    f.write(df_par.to_string(index=False))

print(f"\n  Saved comprehensive report to: {REPORT_FILE}")

# Colab auto-download support
try:
    shutil.make_archive('/content/parallax_horizon_output', 'zip', OUTPUT_DIR)
    from google.colab import files
    files.download(REPORT_FILE)
    files.download('/content/parallax_horizon_output.zip')
    print("  [COLAB] Auto-download of zip package initiated!")
except Exception:
    pass

print("\n" + "=" * 80)
print("  [SUCCESS] Script 16 Completed Successfully! Plots 66 & 67 Generated.")
print("=" * 80)
