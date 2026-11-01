"""
=============================================================
Phase 4c: Dedicated t+60 Solar Specialist Model
Dataset: Kuala Lumpur 1-min GHI + Weather (2025)
DOP: Forecasting of Renewable Resources Using AI
BITS Pilani, Hyderabad | EEE Dept | Prof. Arup Ratan
=============================================================
THE 3 CORE INNOVATIONS:
  1. Single-Task Objective: 100% dedicated to t+60 forecast
     (no loss dilution from t+1 or t+5).
  2. Known Future Astronomical Covariates:
     Feeds GHI_clear(t+60) & Hour(t+60) directly into the fusion head.
     The model knows the exact astronomical solar ceiling 60 mins ahead!
  3. Physics-Constrained Bounded Output:
     Model predicts cloud attenuation factor kt in [0, 1.2]
     GHI_pred(t+60) = kt_pred * GHI_clear(t+60)
     Loss is computed directly on physical W/m2!
=============================================================
"""

import os
import sys
import shutil
import time
import math
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# ── Device Setup ──────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  Device: {device}")
if device.type == 'cuda':
    print(f"  GPU:    {torch.cuda.get_device_name(0)}")

# ── Path Resolution (Works locally & on Colab) ────────────────
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = '/content'

if not os.path.exists(os.path.join(BASE_DIR, 'kuala_lumpur_1min.csv')) and os.path.exists('/content/kuala_lumpur_1min.csv'):
    BASE_DIR = '/content'

DATA_FILE    = os.path.join(BASE_DIR, 'kuala_lumpur_1min.csv')
RESULTS_FILE = os.path.join(BASE_DIR, 'phase4c_results.txt')
PLOTS_DIR    = os.path.join(BASE_DIR, 'phase4c_output')
os.makedirs(PLOTS_DIR, exist_ok=True)

def save_fig(fn):
    p = os.path.join(PLOTS_DIR, fn)
    plt.savefig(p, bbox_inches='tight')
    print(f"  [SAVED] {p}")

plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 150,
    'font.size': 11, 'axes.titlesize': 13,
    'axes.labelsize': 11, 'figure.facecolor': 'white'
})
sns.set_style("whitegrid")
ACCENT = '#FF6B35'; BLUE = '#2C3E50'; CYAN = '#3498DB'
GREEN  = '#2ECC71'; PURPLE = '#9B59B6'; RED = '#E74C3C'

torch.manual_seed(42)
np.random.seed(42)

print("=" * 70)
print("  PHASE 4c: DEDICATED t+60 SOLAR FORECASTER WITH KNOWN FUTURE ASTRONOMY")
print("=" * 70)

# ─────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────
print("\n[1/6] Loading Kuala Lumpur 1-minute dataset...")
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"Cannot find dataset at {DATA_FILE}")

df = pd.read_csv(DATA_FILE)
df['DateTime'] = pd.to_datetime(df['DateTime'])
df['Date']     = df['DateTime'].dt.date
print(f"  Loaded: {len(df):,} records ({df['Date'].nunique()} days)")

# ─────────────────────────────────────────────────────────────
# 2. COMPUTE EXACT ASTRONOMICAL CLEAR-SKY GHI
# ─────────────────────────────────────────────────────────────
print("\n[2/6] Computing astronomical GHI_clear...")
LAT_DEG = 3.1390
LON_DEG = 101.6869

def compute_clearsky(dt_series, lat=LAT_DEG, lon=LON_DEG):
    lat_r = math.radians(lat)
    doy   = dt_series.dt.dayofyear.values
    hour  = (dt_series.dt.hour + dt_series.dt.minute / 60.0).values
    
    B    = 2 * np.pi * (doy - 1) / 365.0
    decl = (0.006918 - 0.399912*np.cos(B) + 0.070257*np.sin(B)
            - 0.006758*np.cos(2*B) + 0.000907*np.sin(2*B))
    EoT  = 229.18 * (0.000075 + 0.001868*np.cos(B) - 0.032077*np.sin(B)
                     - 0.014615*np.cos(2*B) - 0.04089*np.sin(2*B))
    TC   = 4 * (lon - 120.0) + EoT
    H    = np.radians(15 * ((hour + TC / 60.0) - 12.0))
    cos_z = np.clip(np.sin(lat_r)*np.sin(decl) + np.cos(lat_r)*np.cos(decl)*np.cos(H), 0, 1)
    
    E0   = 1 + 0.033 * np.cos(2 * np.pi * doy / 365.0)
    I0   = 1361.0 * E0
    zdeg = np.degrees(np.arccos(cos_z))
    AM   = np.where(cos_z > 0.01,
                    1.0 / (cos_z + 0.50572 * np.power(np.clip(96.07995 - zdeg, 0.01, None), -1.6364)),
                    0.0)
    Gcc  = I0 * cos_z * 0.868 * np.exp(-0.0387 * AM * 2.7 * np.exp(0.01 * AM**1.8))
    return np.clip(Gcc, 0.0, None)

df['GHI_clear'] = compute_clearsky(df['DateTime'])

# Clear-sky index kt = GHI / GHI_clear
df['kt'] = np.where(df['GHI_clear'] > 10.0,
                    np.clip(df['GHI'] / df['GHI_clear'], 0.0, 1.2),
                    0.0)

# Cyclical temporal encodings
doy_v  = df['DateTime'].dt.dayofyear.values
hour_v = (df['DateTime'].dt.hour + df['DateTime'].dt.minute / 60.0).values
df['Hour_sin'] = np.sin(2 * np.pi * hour_v / 24.0)
df['Hour_cos'] = np.cos(2 * np.pi * hour_v / 24.0)
df['DOY_sin']  = np.sin(2 * np.pi * doy_v / 365.0)
df['DOY_cos']  = np.cos(2 * np.pi * doy_v / 365.0)

# 10 Past Sequence Features
SEQ_COLS = [
    'GHI',
    'GHI_clear',
    'kt',
    'Temperature',
    'Relative Humidity',
    'DNI',
    'DHI',
    'Pressure',
    'Hour_sin',
    'Hour_cos'
]

LOOKBACK = 60
HORIZON  = 60  # Dedicated to +60 min!

print(f"  Past sequence features ({len(SEQ_COLS)}): {SEQ_COLS}")
print(f"  Future known features (3): [GHI_clear(t+60), Hour_sin(t+60), Hour_cos(t+60)]")

# ─────────────────────────────────────────────────────────────
# 3. SPLIT & STANDARDIZE
# ─────────────────────────────────────────────────────────────
print("\n[3/6] Splitting train / val / test and normalizing...")
train_mask = df['Month'] <= 8
val_mask   = df['Month'] == 9
test_mask  = df['Month'] >= 10

scaler = StandardScaler()
scaler.fit(df.loc[train_mask, SEQ_COLS])

df_scaled = df.copy()
df_scaled[SEQ_COLS] = scaler.transform(df[SEQ_COLS])

# ─────────────────────────────────────────────────────────────
# 4. SEQUENCE EXTRACTION WITH KNOWN FUTURE COVARIATES
# ─────────────────────────────────────────────────────────────
print("\n[4/6] Building (X_past, X_future, y_actual) sequence tensors...")

seq_mat       = df_scaled[SEQ_COLS].values.astype(np.float32)
ghi_clear_arr = df['GHI_clear'].values.astype(np.float32)
hour_sin_arr  = df['Hour_sin'].values.astype(np.float32)
hour_cos_arr  = df['Hour_cos'].values.astype(np.float32)
ghi_actual    = df['GHI'].values.astype(np.float32)

def extract_t60_dataset(dataframe, stride=1):
    X_past_list   = []
    X_future_list = []
    y_actual_list = []
    y_clear_list  = []
    
    for d, grp in dataframe.groupby('Date'):
        idx = grp.index.values
        n   = len(idx)
        if n < LOOKBACK + HORIZON:
            continue
            
        for t in range(LOOKBACK, n - HORIZON + 1, stride):
            t_future = t + HORIZON - 1   # exactly +60 min
            
            # Filter: Target must be daytime (GHI_clear > 10 W/m2)
            if ghi_clear_arr[idx[t_future]] < 10.0:
                continue
                
            x_past = seq_mat[idx[t - LOOKBACK : t]]   # (60, 10)
            
            # KNOWN FUTURE AT t+60:
            # 1. GHI_clear(t+60) normalized by 1000 W/m2
            # 2. Hour_sin(t+60)
            # 3. Hour_cos(t+60)
            # 4. Astronomical Ramp Rate: GHI_clear(t+60) - GHI_clear(t)
            future_ghi_clr = ghi_clear_arr[idx[t_future]]
            current_ghi_clr= ghi_clear_arr[idx[t - 1]]
            ramp_clr       = (future_ghi_clr - current_ghi_clr) / 1000.0
            
            x_future = [
                future_ghi_clr / 1000.0,
                hour_sin_arr[idx[t_future]],
                hour_cos_arr[idx[t_future]],
                ramp_clr
            ]
            
            y_act = ghi_actual[idx[t_future]]
            y_clr = future_ghi_clr
            
            X_past_list.append(x_past)
            X_future_list.append(x_future)
            y_actual_list.append(y_act)
            y_clear_list.append(y_clr)
            
    return (
        np.array(X_past_list, dtype=np.float32),
        np.array(X_future_list, dtype=np.float32),
        np.array(y_actual_list, dtype=np.float32),
        np.array(y_clear_list, dtype=np.float32)
    )

X_past_tr, X_fut_tr, y_act_tr, y_clr_tr = extract_t60_dataset(df[train_mask], stride=2)
X_past_va, X_fut_va, y_act_va, y_clr_va = extract_t60_dataset(df[val_mask],   stride=1)
X_past_te, X_fut_te, y_act_te, y_clr_te = extract_t60_dataset(df[test_mask],  stride=1)

print(f"  Train: {X_past_tr.shape[0]:,} sequences | Past: {X_past_tr.shape} | Future: {X_fut_tr.shape}")
print(f"  Val:   {X_past_va.shape[0]:,} sequences")
print(f"  Test:  {X_past_te.shape[0]:,} sequences")

BATCH_SIZE = 512
train_ds = TensorDataset(torch.from_numpy(X_past_tr), torch.from_numpy(X_fut_tr),
                         torch.from_numpy(y_act_tr), torch.from_numpy(y_clr_tr))
val_ds   = TensorDataset(torch.from_numpy(X_past_va), torch.from_numpy(X_fut_va),
                         torch.from_numpy(y_act_va), torch.from_numpy(y_clr_va))
test_ds  = TensorDataset(torch.from_numpy(X_past_te), torch.from_numpy(X_fut_te),
                         torch.from_numpy(y_act_te), torch.from_numpy(y_clr_te))

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=(device.type=='cuda'))
val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, pin_memory=(device.type=='cuda'))
test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, pin_memory=(device.type=='cuda'))

# ─────────────────────────────────────────────────────────────
# 5. MODEL ARCHITECTURE: RECURRENT FUSION WITH FUTURE ASTRONOMY
# ─────────────────────────────────────────────────────────────
class DedicatedT60Model(nn.Module):
    """
    Inputs:
      - Past sequence: (B, 60, 10) -> Conv1D + LSTM(64) -> Context vector (B, 64)
      - Known Future astronomy: (B, 4) [GHI_clear(t+60)/1000, Hour_sin, Hour_cos, Ramp]
    
    Fusion Head:
      - Concatenates Context + Future Astronomy -> Dense -> Sigmoid * 1.2
      - Outputs kt(t+60) in range [0, 1.2]
    
    Physics Output:
      - GHI_pred(t+60) = kt_pred * GHI_clear(t+60)
    """
    def __init__(self, n_past_feat=10, n_future_feat=4, hidden_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_past_feat, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.GELU()
        )
        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.15
        )
        # Fusion Head combines past temporal embedding + future known astronomy
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + n_future_feat, 64),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # outputs kt factor in [0, 1]
        )
        
    def forward(self, x_past, x_future, ghi_clear_future):
        # x_past: (B, 60, 10)
        c = self.conv(x_past.permute(0, 2, 1)).permute(0, 2, 1)
        lstm_out, _ = self.lstm(c)
        h_last = lstm_out[:, -1, :]  # (B, 64)
        
        # Concatenate with known future astronomy
        fused = torch.cat([h_last, x_future], dim=1)  # (B, 68)
        
        # Predict cloud factor kt in [0, 1.2] (allow slight cloud edge lens effect)
        kt_pred = self.fusion(fused).squeeze(-1) * 1.2  # (B,)
        
        # Physics Reconstruction
        ghi_pred = kt_pred * ghi_clear_future  # (B,)
        return ghi_pred, kt_pred

model = DedicatedT60Model(n_past_feat=len(SEQ_COLS), n_future_feat=4, hidden_dim=64).to(device)
params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nModel: Dedicated t+60 Recurrent Fusion | Trainable Params: {params:,}")

# ─────────────────────────────────────────────────────────────
# 6. TRAINING (Loss in physical W/m2 normalized by STC 1000 W/m2)
# ─────────────────────────────────────────────────────────────
print("\n[5/6] Training dedicated model for t+60 horizon...")
optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

EPOCHS = 35
best_val = float('inf')
best_state = None
train_losses, val_losses = [], []
t0 = time.time()

for ep in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    for bx_past, bx_fut, by_act, by_clr in train_dl:
        bx_past = bx_past.to(device)
        bx_fut  = bx_fut.to(device)
        by_act  = by_act.to(device)
        by_clr  = by_clr.to(device)
        
        optimizer.zero_grad()
        pred_ghi, _ = model(bx_past, bx_fut, by_clr)
        
        # MSE loss computed on physical GHI (scaled by 1000 W/m2 for clean gradient scale)
        loss = torch.mean(((pred_ghi - by_act) / 1000.0) ** 2)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running_loss += loss.item() * bx_past.size(0)
        
    tr_loss = running_loss / len(train_dl.dataset)
    train_losses.append(tr_loss)
    
    # Validation loop
    model.eval()
    running_val = 0.0
    with torch.no_grad():
        for bx_past, bx_fut, by_act, by_clr in val_dl:
            bx_past = bx_past.to(device)
            bx_fut  = bx_fut.to(device)
            by_act  = by_act.to(device)
            by_clr  = by_clr.to(device)
            
            pred_ghi, _ = model(bx_past, bx_fut, by_clr)
            v_loss = torch.mean(((pred_ghi - by_act) / 1000.0) ** 2)
            running_val += v_loss.item() * bx_past.size(0)
            
    val_loss = running_val / len(val_dl.dataset)
    val_losses.append(val_loss)
    scheduler.step(val_loss)
    
    if val_loss < best_val:
        best_val = val_loss
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
    if ep % 5 == 0 or ep == 1:
        print(f"  Epoch {ep:2d}/{EPOCHS} | Train MSE: {tr_loss:.6f} | Val MSE: {val_loss:.6f} | Best: {best_val:.6f}")

print(f"  Training finished in {time.time()-t0:.1f}s")
model.load_state_dict(best_state)

# ─────────────────────────────────────────────────────────────
# 7. EVALUATION ON TEST SET (Oct–Dec 2025)
# ─────────────────────────────────────────────────────────────
print("\n[6/6] Evaluating on Out-Of-Time Test Set (t+60 min lead time)...")
model.eval()
preds_list = []
with torch.no_grad():
    for bx_past, bx_fut, by_act, by_clr in test_dl:
        bx_past = bx_past.to(device)
        bx_fut  = bx_fut.to(device)
        by_clr  = by_clr.to(device)
        pred_ghi, _ = model(bx_past, bx_fut, by_clr)
        preds_list.append(pred_ghi.cpu().numpy())

y_pred_te = np.concatenate(preds_list)

# Persistence benchmark: GHI(t+60) = GHI(t)
# Unscale current GHI at window end (t-1)
ghi_mean = scaler.mean_[0]
ghi_std  = scaler.scale_[0]
persistence_te = np.clip(X_past_te[:, -1, 0] * ghi_std + ghi_mean, 0, None)

# Calculate final metrics for +60 min
r2_p4c   = r2_score(y_act_te, y_pred_te)
mae_p4c  = mean_absolute_error(y_act_te, y_pred_te)
rmse_p4c = np.sqrt(mean_squared_error(y_act_te, y_pred_te))

p_mae    = mean_absolute_error(y_act_te, persistence_te)
p_rmse   = np.sqrt(mean_squared_error(y_act_te, persistence_te))
skill_p4c= (1.0 - (rmse_p4c / p_rmse)) * 100.0

p3_r2_60   = 0.5266
p3_rmse_60 = 204.67
p3_mae_60  = 155.80
p3_skill_60= 25.79

print("\n" + "=" * 70)
print("  FINAL +60 MINUTE FORECAST COMPARISON (OUT-OF-TIME TEST SET)")
print("=" * 70)
print(f"  Metric              Phase 3 (Multi-Task)   Phase 4c (Dedicated Specialist)   Improvement")
print(f"  ---------------------------------------------------------------------------------------")
print(f"  R² Score:           {p3_r2_60:.4f}                 {r2_p4c:.4f}                            {r2_p4c - p3_r2_60:+.4f}")
print(f"  RMSE (W/m²):        {p3_rmse_60:.2f}                 {rmse_p4c:.2f}                            {rmse_p4c - p3_rmse_60:+.2f} W/m²")
print(f"  MAE (W/m²):         {p3_mae_60:.2f}                 {mae_p4c:.2f}                            {mae_p4c - p3_mae_60:+.2f} W/m²")
print(f"  IEEE Skill Score:   +{p3_skill_60:.2f}%               +{skill_p4c:.2f}%                           {skill_p4c - p3_skill_60:+.2f}%")
print("=" * 70)

# ─────────────────────────────────────────────────────────────
# 8. VISUALIZATIONS
# ─────────────────────────────────────────────────────────────
print("\nGenerating publication-quality comparison figures...")

# Plot 49: Actual vs Predicted Scatter (+60 min)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sample_idx = np.random.choice(len(y_act_te), size=min(8000, len(y_act_te)), replace=False)

axes[0].scatter(y_act_te[sample_idx], y_pred_te[sample_idx],
                alpha=0.25, s=12, color=ACCENT, edgecolors='none')
axes[0].plot([0, 1200], [0, 1200], 'k--', linewidth=2, label='1:1 Ideal Line')
axes[0].set_xlabel('Actual GHI at t+60 (W/m²)', fontweight='bold')
axes[0].set_ylabel('Phase 4c Predicted GHI (W/m²)', fontweight='bold')
axes[0].set_title(f'Phase 4c Dedicated t+60 Scatter\nR² = {r2_p4c:.4f} | RMSE = {rmse_p4c:.2f} W/m²', fontweight='bold')
axes[0].legend()
axes[0].set_xlim(0, 1200); axes[0].set_ylim(0, 1200)

residuals = y_pred_te - y_act_te
sns.histplot(residuals[sample_idx], bins=50, kde=True, color=CYAN, ax=axes[1])
axes[1].axvline(0, color='black', linestyle='--', linewidth=1.5)
axes[1].set_xlabel('Prediction Error Residual (W/m²)', fontweight='bold')
axes[1].set_ylabel('Sample Count', fontweight='bold')
axes[1].set_title(f'Error Distribution (+60 Min Lead Time)\nMean Bias = {np.mean(residuals):.2f} W/m² | Std = {np.std(residuals):.2f} W/m²', fontweight='bold')
plt.tight_layout()
save_fig('49_ph4c_t60_scatter_and_residuals.png')
plt.close()

# Plot 50: Walk-forward Continuous Day Profile (Proof of Noon Descent)
fig, ax = plt.subplots(figsize=(14, 6))
n_show = min(400, len(y_act_te))
time_x = np.arange(n_show)

ax.plot(time_x, y_act_te[:n_show], color=ACCENT, linewidth=2.5, label='Actual Sensor GHI (t+60)')
ax.plot(time_x, y_pred_te[:n_show], color=CYAN, linewidth=2.0, linestyle='--', label='Phase 4c Forecast (t+60)')
ax.plot(time_x, y_clr_te[:n_show], color='green', linewidth=1.5, linestyle=':', label='Known Clear-Sky Ceiling GHI_clear(t+60)')
ax.plot(time_x, persistence_te[:n_show], color='gray', linewidth=1.2, linestyle=':', alpha=0.6, label='Persistence Baseline GHI(t)')

ax.fill_between(time_x, y_act_te[:n_show], y_pred_te[:n_show], color='lightblue', alpha=0.3, label='Forecast Error')
ax.set_title(f'Phase 4c Walk-Forward Continuous Profile (+60 Min Horizon)\n'
             f'Physically Bounded by GHI_clear(t+60) — Zero Runaway Overshoot at Solar Noon', fontweight='bold')
ax.set_xlabel('Consecutive Daytime Minute Indices', fontweight='bold')
ax.set_ylabel('Solar Irradiance (W/m²)', fontweight='bold')
ax.legend(loc='upper right', frameon=True)
plt.tight_layout()
save_fig('50_ph4c_t60_walkforward_day.png')
plt.close()

# Plot 51: Phase 3 vs Phase 4c Benchmark Bar Chart
fig, ax = plt.subplots(figsize=(8, 6))
categories = ['Phase 3 (Multi-Task LSTM)', 'Phase 4c (Dedicated Specialist)']
r2_scores  = [p3_r2_60, r2_p4c]
colors     = [CYAN, ACCENT]

bars = ax.bar(categories, r2_scores, color=colors, width=0.45, edgecolor=BLUE, linewidth=1.5)
for b in bars:
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.015, f"R² = {b.get_height():.4f}",
            ha='center', va='bottom', fontweight='bold', fontsize=12)

ax.set_ylabel('R² Score (+60 Min Horizon)', fontweight='bold')
ax.set_title('Improvement on t+60 Min Forecast Horizon\nPhase 3 Baseline vs Phase 4c Dedicated Model', fontweight='bold')
ax.set_ylim(0, max(r2_scores) + 0.15)
ax.axhline(0.70, color='green', linestyle=':', linewidth=1.5, label='Target Threshold (R² = 0.70)')
ax.legend(loc='lower right')
plt.tight_layout()
save_fig('51_ph4c_t60_benchmark_comparison.png')
plt.close()

# ─────────────────────────────────────────────────────────────
# 9. SAVE DETAILED RESULTS
# ─────────────────────────────────────────────────────────────
with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
    f.write("PHASE 4c: DEDICATED t+60 SOLAR FORECASTING RESULTS (KUALA LUMPUR)\n")
    f.write("DOP: Forecasting of Renewable Resources Using AI\n")
    f.write("BITS Pilani, Hyderabad Campus | EEE Dept | Prof. Arup Ratan\n")
    f.write("=" * 70 + "\n\n")
    f.write("CORE ARCHITECTURAL UPGRADES:\n")
    f.write("  1. Dedicated Single-Horizon Target: +60 minutes lead time only.\n")
    f.write("  2. Known Future Astronomy Covariates: GHI_clear(t+60) and Hour(t+60) fed into fusion head.\n")
    f.write("  3. Physics-based reconstruction: GHI(t+60) = kt_pred * GHI_clear(t+60).\n")
    f.write(f"  4. Trainable Parameters: {params:,} (Regularized compact recurrent fusion).\n\n")
    f.write("METRIC COMPARISON (OUT-OF-TIME TEST SET OCT-DEC 2025):\n")
    f.write(f"  Phase 3 Baseline R² (+60m):   {p3_r2_60:.4f}\n")
    f.write(f"  Phase 4c Model R² (+60m):     {r2_p4c:.4f}  (Improvement: {r2_p4c - p3_r2_60:+.4f})\n")
    f.write(f"  Phase 3 RMSE (+60m):          {p3_rmse_60:.2f} W/m²\n")
    f.write(f"  Phase 4c RMSE (+60m):         {rmse_p4c:.2f} W/m² (Improvement: {rmse_p4c - p3_rmse_60:+.2f} W/m²)\n")
    f.write(f"  Phase 3 MAE (+60m):           {p3_mae_60:.2f} W/m²\n")
    f.write(f"  Phase 4c MAE (+60m):          {mae_p4c:.2f} W/m²  (Improvement: {mae_p4c - p3_mae_60:+.2f} W/m²)\n")
    f.write(f"  Phase 4c Skill Score:         +{skill_p4c:.2f}% over persistence benchmark\n")

print(f"\nSaved detailed results to {RESULTS_FILE}")

# If on Colab, zip and auto-download
try:
    shutil.copy2(RESULTS_FILE, os.path.join(PLOTS_DIR, 'phase4c_results.txt'))
    shutil.make_archive('/content/phase4c_output', 'zip', PLOTS_DIR)
    from google.colab import files
    files.download('/content/phase4c_results.txt')
    files.download('/content/phase4c_output.zip')
    print("Colab auto-download initiated!")
except Exception:
    pass

print("\n" + "=" * 70)
print("  [DONE] Phase 4c Completed Successfully!")
print("  Generated Plots: 49, 50, 51")
print("=" * 70)

