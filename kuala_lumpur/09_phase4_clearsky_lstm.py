"""
=============================================================
Phase 4: Clear-Sky Index LSTM + Conv1D + Attention
Dataset: Kuala Lumpur 1-min GHI + Weather (2025)
DOP: Forecasting of Renewable Resources Using AI
BITS Pilani, Hyderabad | EEE Dept | Prof. Arup Ratan
=============================================================
PROBLEM WITH PHASE 3:
  Raw GHI LSTM sees a rising trend from 10:30–11:30 and
  extrapolates linearly → over-predicts after solar noon.
  At t+60 R² = 0.53.

PHASE 4 APPROACH:
  1. Clear-Sky Index  k_t = GHI / GHI_clear
     - GHI_clear from simplified Kasten-Young formula
     - k_t ∈ [0,1] captures only cloud attenuation
     - Removes diurnal bell curve → LSTM cannot overshoot noon
  2. Architecture upgrade: Conv1D feature extractor +
     2-Layer BiLSTM + Temporal Attention + Linear head
  3. 10 features per minute (k_t replaces raw GHI + extras)
  4. 120-epoch training with CosineAnnealing LR
  5. Weighted loss: heavier penalty on t+60 horizon
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
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# ── Paths & Setup ─────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATA_FILE     = os.path.join(BASE_DIR, 'kuala_lumpur_1min.csv')
RESULTS_FILE  = os.path.join(BASE_DIR, 'phase4_results.txt')
PLOTS_DIR     = os.path.join(BASE_DIR, 'plots')
ROOT_PLOTS    = os.path.abspath(os.path.join(BASE_DIR, '..', 'plots'))
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(ROOT_PLOTS, exist_ok=True)

def save_fig(filename):
    lp = os.path.join(PLOTS_DIR, filename)
    rp = os.path.join(ROOT_PLOTS, filename)
    plt.savefig(lp, bbox_inches='tight')
    shutil.copy2(lp, rp)
    print(f"  [SAVED] {lp}")

plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 150,
    'font.size': 11, 'axes.titlesize': 13,
    'axes.labelsize': 11, 'figure.facecolor': 'white'
})
sns.set_style("whitegrid")

ACCENT = '#FF6B35'; BLUE = '#2C3E50'; CYAN = '#3498DB'
GREEN  = '#2ECC71'; PURPLE = '#9B59B6'; RED = '#E74C3C'

print("=" * 70)
print("  PHASE 4: CLEARSKY-INDEX LSTM — KUALA LUMPUR SOLAR FORECASTING")
print("  Architecture: Conv1D + BiLSTM + Attention | k_t Normalization")
print("=" * 70)

torch.manual_seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────────────────────
# SECTION 1: LOAD DATA
# ─────────────────────────────────────────────────────────────
print("\n[1/7] Loading 1-minute Kuala Lumpur dataset...")
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

df = pd.read_csv(DATA_FILE)
df['DateTime'] = pd.to_datetime(df['DateTime'])
df['Date']     = df['DateTime'].dt.date
print(f"  Loaded: {len(df):,} rows | {df['Date'].nunique()} days")
print(f"  Columns: {df.columns.tolist()}")

# ─────────────────────────────────────────────────────────────
# SECTION 2: COMPUTE CLEAR-SKY GHI (Kasten-Young Approximation)
# ─────────────────────────────────────────────────────────────
print("\n[2/7] Computing Clear-Sky GHI (Kasten-Young) and k_t index...")

# Kuala Lumpur coordinates
LAT_DEG = 3.1390
LON_DEG = 101.6869

def compute_clearsky_ghi(datetime_series, lat_deg=LAT_DEG):
    """
    Simplified Kasten-Young clear-sky model.
    Returns GHI_clear in W/m² for each timestamp.
    
    Steps:
      1. Solar declination δ from day-of-year
      2. Hour angle H from solar time
      3. cos(zenith) = sin(lat)sin(δ) + cos(lat)cos(δ)cos(H)
      4. Air mass AM = 1 / (cos(z) + 0.50572*(96.07995 - z_deg)^-1.6364)
      5. GHI_clear = I0 * cos(z) * exp(-0.8662 * TL * AM * delta_R)
         (simplified: TL=2.0, delta_R=1.0 for tropical)
    """
    lat_rad = math.radians(lat_deg)
    doy     = datetime_series.dt.dayofyear.values
    hour    = (datetime_series.dt.hour +
               datetime_series.dt.minute / 60.0).values

    # Solar declination (radians)
    B       = 2 * np.pi * (doy - 1) / 365.0
    decl    = (0.006918 - 0.399912 * np.cos(B) + 0.070257 * np.sin(B)
               - 0.006758 * np.cos(2*B) + 0.000907 * np.sin(2*B))

    # Equation of Time (minutes)
    EoT     = (229.18 * (0.000075 + 0.001868 * np.cos(B)
               - 0.032077 * np.sin(B) - 0.014615 * np.cos(2*B)
               - 0.04089 * np.sin(2*B)))

    # Local Standard Time Meridian for UTC+8 (KL)
    LSTM_deg = 120.0
    TC       = 4 * (LON_DEG - LSTM_deg) + EoT          # minutes
    solar_time = hour + TC / 60.0                        # hours
    H       = np.radians(15 * (solar_time - 12.0))      # hour angle

    # Cosine of solar zenith angle
    cos_z = (np.sin(lat_rad) * np.sin(decl)
             + np.cos(lat_rad) * np.cos(decl) * np.cos(H))
    cos_z = np.clip(cos_z, 0.0, 1.0)                    # nighttime → 0

    # Extraterrestrial irradiance (corrected for eccentricity)
    E0 = 1 + 0.033 * np.cos(2 * np.pi * doy / 365.0)
    I0 = 1361.0 * E0                                     # W/m²

    # Air mass (Kasten-Young 1989) — only when sun is up
    z_deg = np.degrees(np.arccos(cos_z))
    AM    = np.where(
        cos_z > 0.01,
        1.0 / (cos_z + 0.50572 * np.power(np.clip(96.07995 - z_deg, 0.01, None), -1.6364)),
        0.0
    )

    # Ineichen-Perez simplified: Linke Turbidity TL=2.7 (tropical)
    TL   = 2.7
    fh1  = np.exp(-0.0 / 8000.0)     # altitude = 0 m (sea level)
    fh2  = np.exp(-0.0 / 1250.0)
    cg1  = 5.09e-5 * 0.0 + 0.868    # altitude correction
    cg2  = 3.92e-5 * 0.0 + 0.0387
    GHI_clear = I0 * cos_z * cg1 * np.exp(
        -cg2 * AM * (fh1 + fh2 * (TL - 1.0)) * np.exp(0.01 * np.power(AM, 1.8))
    )
    GHI_clear = np.clip(GHI_clear, 0.0, None)
    return GHI_clear

df['GHI_clear'] = compute_clearsky_ghi(df['DateTime'])

# Clear-sky index  k_t = GHI / GHI_clear,  clipped to [0, 1.2]
# k_t > 1 can happen under lens effect of broken cloud edges
EPSILON = 5.0   # avoid division by zero at night (below 5 W/m²)
df['kt'] = np.where(
    df['GHI_clear'] > EPSILON,
    np.clip(df['GHI'] / df['GHI_clear'], 0.0, 1.2),
    0.0
)

# Daytime mask for evaluation (GHI_clear > 20 W/m²)
daytime_mask_col = (df['GHI_clear'] > 20.0)

print(f"  GHI_clear range: {df['GHI_clear'].min():.1f} – {df['GHI_clear'].max():.1f} W/m²")
print(f"  k_t range:       {df['kt'].min():.3f} – {df['kt'].max():.3f}")
print(f"  Daytime samples: {daytime_mask_col.sum():,} / {len(df):,}")

# ─────────────────────────────────────────────────────────────
# SECTION 3: BUILD ENHANCED FEATURE SET (10 features)
# ─────────────────────────────────────────────────────────────
print("\n[3/7] Engineering 10-feature input set...")

# Day-of-year cyclical encoding
doy_arr = df['DateTime'].dt.dayofyear.values
df['DOY_sin'] = np.sin(2 * np.pi * doy_arr / 365.0)
df['DOY_cos'] = np.cos(2 * np.pi * doy_arr / 365.0)

# Re-create hour cyclical in case column name varies
hour_arr = (df['DateTime'].dt.hour + df['DateTime'].dt.minute / 60.0).values
df['Hour_sin_p4'] = np.sin(2 * np.pi * hour_arr / 24.0)
df['Hour_cos_p4'] = np.cos(2 * np.pi * hour_arr / 24.0)

# Delta-k_t: 1-minute change in clear-sky index (cloud ramp proxy)
df['delta_kt'] = df['kt'].diff().fillna(0.0).clip(-0.5, 0.5)

FEATURE_COLS = [
    'kt',               #  1. Clear-Sky Index (target-relative, [0,1.2])
    'Temperature',      #  2. Ambient Temperature (°C)
    'Relative Humidity',#  3. Relative Humidity (%)
    'DNI',              #  4. Direct Normal Irradiance (W/m²)
    'DHI',              #  5. Diffuse Horizontal Irradiance (W/m²)
    'Pressure',         #  6. Atmospheric Pressure (hPa)
    'Hour_sin_p4',      #  7. Hour-of-day sine (diurnal cycle)
    'Hour_cos_p4',      #  8. Hour-of-day cosine
    'DOY_sin',          #  9. Day-of-year sine (seasonal)
    'DOY_cos',          # 10. Day-of-year cosine
    'delta_kt',         # 11. 1-min change in k_t (cloud ramp rate proxy)
]
TARGET_COL = 'kt'       # Predict k_t; reconstruct GHI = k_t * GHI_clear

N_FEATURES = len(FEATURE_COLS)
LOOKBACK   = 60
HORIZONS   = [1, 5, 15, 30, 60]

print(f"  Feature set ({N_FEATURES}): {FEATURE_COLS}")
print(f"  Target: '{TARGET_COL}' | Lookback: {LOOKBACK} min | Horizons: {HORIZONS} min")

# ─────────────────────────────────────────────────────────────
# SECTION 4: TRAIN / VAL / TEST SPLIT + SCALING
# ─────────────────────────────────────────────────────────────
print("\n[4/7] Splitting and scaling datasets...")
train_mask = df['Month'] <= 8
val_mask   = df['Month'] == 9
test_mask  = df['Month'] >= 10

print(f"  Train: {train_mask.sum():,} rows (Jan–Aug 2025)")
print(f"  Val:   {val_mask.sum():,} rows (Sep 2025)")
print(f"  Test:  {test_mask.sum():,} rows (Oct–Dec 2025)")

# Fit scaler ONLY on training data (no look-ahead bias)
feature_scaler = StandardScaler()
feature_scaler.fit(df.loc[train_mask, FEATURE_COLS])
df_scaled = df.copy()
df_scaled[FEATURE_COLS] = feature_scaler.transform(df[FEATURE_COLS])

# k_t target scaler (fit on training daytime values)
kt_mean = df.loc[train_mask & (df['GHI_clear'] > EPSILON), 'kt'].mean()
kt_std  = df.loc[train_mask & (df['GHI_clear'] > EPSILON), 'kt'].std()
print(f"  k_t train mean={kt_mean:.4f}, std={kt_std:.4f}")

# Manual standardize kt (already in FEATURE_COLS; reuse feature_scaler idx 0)
kt_idx = FEATURE_COLS.index('kt')
scaled_kt = df_scaled[FEATURE_COLS].values[:, kt_idx]   # already scaled

# ─────────────────────────────────────────────────────────────
# SECTION 5: SEQUENCE EXTRACTION
# ─────────────────────────────────────────────────────────────
print("\n[5/7] Extracting 60-minute sliding window sequences (daytime only)...")

def extract_sequences(dataframe, scaled_feat_arr, scaled_kt_arr,
                      ghi_clear_arr, ghi_raw_arr,
                      max_horizon=60, stride=1, daytime_only=False):
    """
    Build (X, y_kt_scaled, y_ghi_clear_future, y_ghi_actual) arrays.
    daytime_only: skip windows where the TARGET step is at night.
    """
    X_list, ykt_list, ycl_list, yact_list = [], [], [], []

    for d, group in dataframe.groupby('Date'):
        idx = group.index.values
        n   = len(idx)
        if n < LOOKBACK + max_horizon:
            continue

        feats      = scaled_feat_arr[idx]
        kt_tgt     = scaled_kt_arr[idx]
        ghi_clr    = ghi_clear_arr[idx]
        ghi_act    = ghi_raw_arr[idx]

        for t in range(LOOKBACK, n - max_horizon + 1, stride):
            # Filter: the t+60 target must be daytime
            if daytime_only and ghi_clr[t + max_horizon - 1] < EPSILON:
                continue

            x_seq  = feats[t - LOOKBACK : t]                   # (60, N_FEAT)

            y_kt_s = [                                          # scaled k_t targets
                kt_tgt[t],
                kt_tgt[t + 4],
                kt_tgt[t + 14],
                kt_tgt[t + 29],
                kt_tgt[t + 59],
            ]
            y_cl = [                                            # GHI_clear at future steps
                ghi_clr[t],
                ghi_clr[t + 4],
                ghi_clr[t + 14],
                ghi_clr[t + 29],
                ghi_clr[t + 59],
            ]
            y_ac = [                                            # actual GHI at future steps
                ghi_act[t],
                ghi_act[t + 4],
                ghi_act[t + 14],
                ghi_act[t + 29],
                ghi_act[t + 59],
            ]
            X_list.append(x_seq)
            ykt_list.append(y_kt_s)
            ycl_list.append(y_cl)
            yact_list.append(y_ac)

    return (
        np.array(X_list,   dtype=np.float32),
        np.array(ykt_list, dtype=np.float32),
        np.array(ycl_list, dtype=np.float32),
        np.array(yact_list,dtype=np.float32),
    )

feat_arr      = df_scaled[FEATURE_COLS].values
kt_arr        = feat_arr[:, kt_idx]
ghi_clear_arr = df['GHI_clear'].values
ghi_raw_arr   = df['GHI'].values

# Stride=3 on train to keep CPU runtime manageable
X_tr, ykt_tr, ycl_tr, yact_tr = extract_sequences(
    df[train_mask], feat_arr, kt_arr, ghi_clear_arr, ghi_raw_arr, stride=3)
X_va, ykt_va, ycl_va, yact_va = extract_sequences(
    df[val_mask],   feat_arr, kt_arr, ghi_clear_arr, ghi_raw_arr, stride=2)
X_te, ykt_te, ycl_te, yact_te = extract_sequences(
    df[test_mask],  feat_arr, kt_arr, ghi_clear_arr, ghi_raw_arr, stride=1)

print(f"  Train: X={X_tr.shape}  y_kt={ykt_tr.shape}")
print(f"  Val:   X={X_va.shape}  y_kt={ykt_va.shape}")
print(f"  Test:  X={X_te.shape}  y_kt={ykt_te.shape}")

BATCH_SIZE = 512
train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(ykt_tr))
val_ds   = TensorDataset(torch.from_numpy(X_va), torch.from_numpy(ykt_va))
test_ds  = TensorDataset(torch.from_numpy(X_te), torch.from_numpy(ykt_te))
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ─────────────────────────────────────────────────────────────
# SECTION 6: MODEL — Conv1D + BiLSTM + Temporal Attention
# ─────────────────────────────────────────────────────────────
class TemporalAttention(nn.Module):
    """Single-head additive attention over LSTM time steps."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn_w = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_out):
        # lstm_out: (B, T, H)
        scores  = self.attn_w(lstm_out).squeeze(-1)   # (B, T)
        weights = torch.softmax(scores, dim=-1)        # (B, T)
        context = (weights.unsqueeze(-1) * lstm_out).sum(dim=1)  # (B, H)
        return context, weights


class Phase4SolarModel(nn.Module):
    """
    Conv1D  →  2-Layer Bidirectional LSTM  →  Attention  →  Linear head
    Input:  (B, 60, 10)
    Output: (B, 5)   — scaled k_t at horizons [1,5,15,30,60]
    """
    def __init__(self, n_feat=10, conv_ch=32, hidden_dim=96,
                 n_layers=2, n_out=5, dropout=0.25):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_feat, conv_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_ch),
            nn.GELU(),
            nn.Conv1d(conv_ch, conv_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_ch),
            nn.GELU(),
        )
        self.lstm = nn.LSTM(
            input_size=conv_ch,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        bi_dim = hidden_dim * 2  # bidirectional doubles hidden dim
        self.attn = TemporalAttention(bi_dim)
        self.head  = nn.Sequential(
            nn.LayerNorm(bi_dim),
            nn.Linear(bi_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, n_out),
        )

    def forward(self, x):
        # x: (B, T, F) → Conv1d expects (B, F, T)
        c = self.conv(x.permute(0, 2, 1))       # (B, conv_ch, T)
        c = c.permute(0, 2, 1)                  # (B, T, conv_ch)
        lstm_out, _ = self.lstm(c)              # (B, T, 2*H)
        ctx, _      = self.attn(lstm_out)       # (B, 2*H)
        return self.head(ctx)                   # (B, 5)


model = Phase4SolarModel(n_feat=N_FEATURES, conv_ch=32,
                         hidden_dim=96, n_layers=2, n_out=len(HORIZONS))

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nModel Architecture:\n{model}")
print(f"Total Trainable Parameters: {total_params:,}")

# ─────────────────────────────────────────────────────────────
# SECTION 7: TRAINING
# ─────────────────────────────────────────────────────────────
print("\n[6/7] Training Phase 4 model...")

# Horizon weights: penalise longer horizons more (t+60 gets 3x weight)
HORIZON_WEIGHTS = torch.tensor([1.0, 1.5, 2.0, 2.5, 3.0])

def weighted_mse(pred, target, weights=HORIZON_WEIGHTS):
    sq_err = (pred - target) ** 2           # (B, 5)
    w_err  = sq_err * weights.unsqueeze(0)  # broadcast over batch
    return w_err.mean()

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
EPOCHS    = 40
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS, eta_min=1e-5)

best_val   = float('inf')
best_state = None
train_hist = []
val_hist   = []

t0 = time.time()
for epoch in range(1, EPOCHS + 1):
    # ── Train ─────────────────────────────────────────────────
    model.train()
    tr_loss = 0.0
    for bx, by in train_dl:
        optimizer.zero_grad()
        out  = model(bx)
        loss = weighted_mse(out, by)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tr_loss += loss.item() * bx.size(0)
    tr_loss /= len(train_ds)
    train_hist.append(tr_loss)

    # ── Validate ───────────────────────────────────────────────
    model.eval()
    va_loss = 0.0
    with torch.no_grad():
        for bx, by in val_dl:
            out     = model(bx)
            va_loss += weighted_mse(out, by).item() * bx.size(0)
    va_loss /= len(val_ds)
    val_hist.append(va_loss)
    scheduler.step()

    if va_loss < best_val:
        best_val   = va_loss
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if epoch % 5 == 0 or epoch == 1:
        lr_now = optimizer.param_groups[0]['lr']
        print(f"  Epoch {epoch:3d}/{EPOCHS} | "
              f"TrainLoss={tr_loss:.5f} | ValLoss={va_loss:.5f} | "
              f"Best={best_val:.5f} | LR={lr_now:.2e}")

print(f"  Training time: {time.time()-t0:.1f}s")
model.load_state_dict(best_state)

# ─────────────────────────────────────────────────────────────
# SECTION 8: EVALUATION (k_t → GHI reconstruction)
# ─────────────────────────────────────────────────────────────
print("\n[7/7] Evaluating on out-of-time test set (Oct–Dec 2025)...")

model.eval()
kt_pred_scaled_list = []
with torch.no_grad():
    for bx, _ in test_dl:
        kt_pred_scaled_list.append(model(bx).numpy())
kt_pred_scaled = np.vstack(kt_pred_scaled_list)   # (N_test, 5)

# ── Recover physical units ─────────────────────────────────────
#   k_t was scaled by the feature_scaler at index kt_idx
kt_col_mean = feature_scaler.mean_[kt_idx]
kt_col_std  = feature_scaler.scale_[kt_idx]

kt_pred     = kt_pred_scaled * kt_col_std + kt_col_mean   # unscaled k_t
kt_actual   = ykt_te         * kt_col_std + kt_col_mean

# Reconstruct GHI from k_t and clear-sky
# yact_te already holds actual GHI_raw, ycl_te holds GHI_clear
ghi_pred_wm2   = np.clip(kt_pred   * ycl_te, 0, None)   # k_t * GHI_clear
ghi_actual_wm2 = np.clip(yact_te,             0, None)

# Persistence baseline: last observed GHI in the lookback window
#   X_te[:, -1, 0] = scaled kt at last step; reconstruct GHI
kt_persist_scaled = X_te[:, -1, kt_idx]
kt_persist        = kt_persist_scaled * kt_col_std + kt_col_mean
# GHI_clear at time t (first column of ycl_te → +1 min ≈ same)
ghi_persist       = np.clip(kt_persist[:, np.newaxis] * ycl_te, 0, None)

# ── Compute metrics per horizon ────────────────────────────────
metrics  = {}
p3_r2    = [0.8978, 0.7494, 0.6321, 0.5712, 0.5266]   # Phase 3 baseline
p3_rmse  = [91.22,  142.99, 173.90, 189.50, 204.67]

print(f"\n  {'Horizon':<10} {'R² Ph3':>8} {'R² Ph4':>8} {'Δ R²':>8}  "
      f"{'RMSE Ph3':>10} {'RMSE Ph4':>10}  {'Skill%':>8}")
print("  " + "-" * 74)

for i, h in enumerate(HORIZONS):
    act_h  = ghi_actual_wm2[:, i]
    pred_h = ghi_pred_wm2[:, i]
    pers_h = ghi_persist[:, i]

    # Mask: only evaluate where the target is daytime
    day_m = ycl_te[:, i] > EPSILON
    if day_m.sum() < 100:
        day_m = np.ones(len(act_h), dtype=bool)

    r2    = r2_score(act_h[day_m], pred_h[day_m])
    mae   = mean_absolute_error(act_h[day_m], pred_h[day_m])
    rmse  = np.sqrt(mean_squared_error(act_h[day_m], pred_h[day_m]))
    p_rmse= np.sqrt(mean_squared_error(act_h[day_m], pers_h[day_m]))
    skill = (1.0 - rmse / p_rmse) * 100.0

    metrics[h] = dict(R2=r2, MAE=mae, RMSE=rmse, P_RMSE=p_rmse, Skill=skill)

    delta_r2 = r2 - p3_r2[i]
    print(f"  +{h:<9d} {p3_r2[i]:>8.4f} {r2:>8.4f} {delta_r2:>+8.4f}  "
          f"{p3_rmse[i]:>10.2f} {rmse:>10.2f}  {skill:>+8.2f}%")

# ─────────────────────────────────────────────────────────────
# SECTION 9: VISUALIZATIONS
# ─────────────────────────────────────────────────────────────
print("\nGenerating Phase 4 plots...")

# ── Plot 40: Training Convergence ─────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
ep_arr  = range(1, EPOCHS + 1)
ax.plot(ep_arr, train_hist, 'o-', color=CYAN, linewidth=2, ms=4, label='Train Loss (Weighted MSE)')
ax.plot(ep_arr, val_hist,   's--', color=ACCENT, linewidth=2, ms=4, label='Val Loss (Weighted MSE)')
ax.set_xlabel('Epoch', fontweight='bold')
ax.set_ylabel('Weighted MSE Loss (Normalized k_t space)', fontweight='bold')
ax.set_title('Phase 4 — Conv1D + BiLSTM + Attention Training Convergence\n'
             'Clear-Sky Index k_t Target | Cosine LR Annealing | 40 Epochs',
             fontweight='bold')
ax.legend()
plt.tight_layout()
save_fig('40_ph4_training_convergence.png')
plt.close()

# ── Plot 41: Phase 3 vs Phase 4 — R² Comparison bar chart ────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

horizon_labels = [f'+{h}m' for h in HORIZONS]
r2_ph3 = p3_r2
r2_ph4 = [metrics[h]['R2'] for h in HORIZONS]

x     = np.arange(len(HORIZONS))
width = 0.35

bars3 = axes[0].bar(x - width/2, r2_ph3, width, label='Phase 3 (Raw GHI LSTM)',
                    color=CYAN, edgecolor=BLUE, alpha=0.8)
bars4 = axes[0].bar(x + width/2, r2_ph4, width, label='Phase 4 (k_t Conv+BiLSTM)',
                    color=ACCENT, edgecolor='#8B3A0C', alpha=0.9)

for b in bars3:
    axes[0].text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                 f'{b.get_height():.3f}', ha='center', va='bottom', fontsize=8)
for b in bars4:
    axes[0].text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                 f'{b.get_height():.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

axes[0].set_xticks(x)
axes[0].set_xticklabels(horizon_labels)
axes[0].set_xlabel('Forecast Horizon', fontweight='bold')
axes[0].set_ylabel('R² Score', fontweight='bold')
axes[0].set_title('Phase 3 vs Phase 4 — R² Score Comparison\n'
                  'Across All Forecast Horizons', fontweight='bold')
axes[0].legend()
axes[0].set_ylim(0, 1.05)
axes[0].axhline(0.8, color='green', linestyle=':', linewidth=1.2, label='R²=0.80 target')

# RMSE comparison
rmse_ph4 = [metrics[h]['RMSE'] for h in HORIZONS]
axes[1].plot(HORIZONS, p3_rmse, 'o-', color=CYAN, linewidth=2.5,
             ms=7, label='Phase 3 RMSE (W/m²)')
axes[1].plot(HORIZONS, rmse_ph4, 's-', color=ACCENT, linewidth=2.5,
             ms=7, label='Phase 4 RMSE (W/m²)')
axes[1].fill_between(HORIZONS, p3_rmse, rmse_ph4,
                     where=[r4 < r3 for r3, r4 in zip(p3_rmse, rmse_ph4)],
                     alpha=0.2, color='green', label='Improvement')
axes[1].fill_between(HORIZONS, p3_rmse, rmse_ph4,
                     where=[r4 >= r3 for r3, r4 in zip(p3_rmse, rmse_ph4)],
                     alpha=0.2, color='red', label='Regression')
axes[1].set_xlabel('Forecast Horizon (minutes)', fontweight='bold')
axes[1].set_ylabel('RMSE (W/m²)', fontweight='bold')
axes[1].set_title('Phase 3 vs Phase 4 — RMSE Comparison\n'
                  'Lower is better', fontweight='bold')
axes[1].set_xticks(HORIZONS)
axes[1].legend()
plt.tight_layout()
save_fig('41_ph4_vs_ph3_comparison.png')
plt.close()

# ── Plot 42: Actual vs Predicted GHI (t+60) scatter ──────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

act_60  = ghi_actual_wm2[:, 4]
pred_60 = ghi_pred_wm2[:, 4]
day_60  = ycl_te[:, 4] > EPSILON
sid     = np.random.choice(np.where(day_60)[0],
                            size=min(8000, day_60.sum()), replace=False)

axes[0].scatter(act_60[sid], pred_60[sid],
                alpha=0.3, s=10, color=ACCENT, edgecolors='none')
axes[0].plot([0, 1200], [0, 1200], 'k--', lw=2, label='1:1 Ideal Fit')
axes[0].set_xlabel('Actual GHI (W/m²)', fontweight='bold')
axes[0].set_ylabel('Predicted GHI (W/m²)', fontweight='bold')
axes[0].set_title(f'Phase 4 — GHI Scatter (+60 Min Horizon)\n'
                  f'R² = {metrics[60]["R2"]:.4f} | RMSE = {metrics[60]["RMSE"]:.2f} W/m²',
                  fontweight='bold')
axes[0].legend()
axes[0].set_xlim(0, 1200); axes[0].set_ylim(0, 1200)

# Residuals at t+60
resid_60 = pred_60[sid] - act_60[sid]
sns.histplot(resid_60, bins=60, kde=True, color=ACCENT, ax=axes[1])
axes[1].axvline(0, color='k', ls='--', lw=1.5)
axes[1].set_xlabel('Prediction Error (W/m²)', fontweight='bold')
axes[1].set_ylabel('Count', fontweight='bold')
axes[1].set_title(f'+60 Min Forecast Error Distribution\n'
                  f'Mean Bias = {np.mean(resid_60):.2f} W/m² | Std = {np.std(resid_60):.2f} W/m²',
                  fontweight='bold')
plt.tight_layout()
save_fig('42_ph4_60min_scatter_residuals.png')
plt.close()

# ── Plot 43: Walk-forward continuous segment (t+1 and t+60 overlaid) ─
# Use the first 400 contiguous test samples for a clean day-like profile
fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=False)

for ax_i, h_idx in enumerate([0, 4]):
    h      = HORIZONS[h_idx]
    n_show = min(400, ghi_actual_wm2.shape[0])
    t_arr  = np.arange(n_show)
    act_h  = ghi_actual_wm2[:n_show, h_idx]
    pred_h = ghi_pred_wm2[:n_show, h_idx]
    pers_h = ghi_persist[:n_show, h_idx]

    axes[ax_i].plot(t_arr, act_h,  color=ACCENT, lw=2,   label='Actual GHI')
    axes[ax_i].plot(t_arr, pred_h, color=CYAN,   lw=2,   ls='--', label=f'Phase 4 LSTM (+{h}m)')
    axes[ax_i].plot(t_arr, pers_h, color='gray', lw=1.2, ls=':',  label='Persistence', alpha=0.7)
    axes[ax_i].fill_between(t_arr, act_h, pred_h, color='lightblue', alpha=0.25)

    r2_seg  = r2_score(act_h, pred_h)
    mae_seg = mean_absolute_error(act_h, pred_h)
    axes[ax_i].set_title(f'Phase 4 Walk-Forward Forecast — +{h} Min Horizon  '
                         f'(R²={r2_seg:.4f} | MAE={mae_seg:.1f} W/m²)',
                         fontweight='bold')
    axes[ax_i].set_ylabel('GHI (W/m²)', fontweight='bold')
    axes[ax_i].legend(loc='upper right')

axes[1].set_xlabel('Sample Index (consecutive test minutes)', fontweight='bold')
plt.suptitle('Phase 4 — k_t Conv1D+BiLSTM+Attention Walk-Forward Solar Forecast\n'
             'Kuala Lumpur 2025 (Oct–Dec Out-of-Time Test)', fontweight='bold', y=1.01)
plt.tight_layout()
save_fig('43_ph4_walkforward_t1_t60.png')
plt.close()

# ── Plot 44: Skill Score comparison Phase 3 vs Phase 4 ────────
fig, ax = plt.subplots(figsize=(9, 5))
skill_ph4 = [metrics[h]['Skill'] for h in HORIZONS]

ax.plot(HORIZONS, skill_ph4, 's-',  color=ACCENT, lw=2.5, ms=8, label='Phase 4 Skill Score')
skill_ph3 = [3.07, 11.66, 17.16, 20.30, 25.79]
ax.plot(HORIZONS, skill_ph3, 'o--', color=CYAN,   lw=2.0, ms=7, label='Phase 3 Skill Score')
ax.fill_between(HORIZONS, skill_ph3, skill_ph4,
                where=[s4 > s3 for s3, s4 in zip(skill_ph3, skill_ph4)],
                color='green', alpha=0.2, label='Phase 4 gain')
ax.fill_between(HORIZONS, skill_ph3, skill_ph4,
                where=[s4 <= s3 for s3, s4 in zip(skill_ph3, skill_ph4)],
                color='red', alpha=0.2, label='Phase 3 better')
ax.set_xlabel('Forecast Horizon (minutes)', fontweight='bold')
ax.set_ylabel('IEEE Forecast Skill Score over Persistence (%)', fontweight='bold')
ax.set_title('Phase 4 vs Phase 3 — Forecast Skill Score Improvement\n'
             'SS = [1 − RMSE_model/RMSE_persistence] × 100', fontweight='bold')
ax.set_xticks(HORIZONS)
ax.legend()
plt.tight_layout()
save_fig('44_ph4_skill_score_comparison.png')
plt.close()

# ─────────────────────────────────────────────────────────────
# SECTION 10: WRITE RESULTS
# ─────────────────────────────────────────────────────────────
with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
    f.write("PHASE 4: CLEAR-SKY INDEX LSTM SOLAR FORECASTING RESULTS\n")
    f.write("DOP: Forecasting of Renewable Resources Using AI\n")
    f.write("BITS Pilani, Hyderabad | EEE Dept | Prof. Arup Ratan\n")
    f.write("=" * 70 + "\n\n")

    f.write("APPROACH:\n")
    f.write("  - Target: k_t = GHI / GHI_clear  (Clear-Sky Index)\n")
    f.write("  - GHI_clear: Ineichen-Perez simplified (Kasten-Young AM)\n")
    f.write("  - Architecture: Conv1D(32) + 2xBiLSTM(96) + Attention + Linear\n")
    f.write("  - 10 features per minute: k_t, T, RH, DNI, DHI, P, Hr_sin, Hr_cos, DOY_sin, DOY_cos\n")
    f.write(f"  - Lookback: {LOOKBACK} min  |  Horizons: {HORIZONS} min\n")
    f.write("  - Training: 40 epochs, AdamW, CosineAnnealing LR, Weighted MSE\n")
    f.write(f"  - Trainable params: {total_params:,}\n\n")

    f.write("KEY INSIGHT — WHY k_t WORKS:\n")
    f.write("  Raw GHI LSTM saw rising trend 10:30→11:30 and extrapolated\n")
    f.write("  linearly past solar noon → over-predicted at +60 min.\n")
    f.write("  k_t = GHI/GHI_clear removes the deterministic bell curve.\n")
    f.write("  The LSTM now models ONLY cloud attenuation (stochastic part).\n")
    f.write("  Prediction: GHI_pred = k_t_pred × GHI_clear(future astronomy)\n\n")

    f.write(f"{'Horizon':<10} {'R²(Ph3)':<10} {'R²(Ph4)':<10} {'ΔR²':<10} "
            f"{'RMSE_Ph3':<12} {'RMSE_Ph4':<12} {'MAE':<10} {'Skill%':<10}\n")
    f.write("-" * 80 + "\n")
    for i, h in enumerate(HORIZONS):
        m   = metrics[h]
        dr2 = m['R2'] - p3_r2[i]
        f.write(f"+{h:<9d} {p3_r2[i]:<10.4f} {m['R2']:<10.4f} {dr2:<+10.4f} "
                f"{p3_rmse[i]:<12.2f} {m['RMSE']:<12.2f} {m['MAE']:<10.2f} {m['Skill']:<+10.2f}%\n")

print(f"\n  Results saved to {RESULTS_FILE}")

print("\n" + "=" * 70)
print("  [DONE] Phase 4 Complete!")
print("  Plots generated: 40, 41, 42, 43, 44")
print("=" * 70)
