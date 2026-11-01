"""
=============================================================
Phase 4: Clear-Sky Index LSTM + Conv1D + Attention
COLAB VERSION — GPU-enabled, no __file__ dependency
Dataset: Kuala Lumpur 1-min GHI + Weather (2025)
DOP: Forecasting of Renewable Resources Using AI
BITS Pilani, Hyderabad | EEE Dept | Prof. Arup Ratan
=============================================================
HOW TO USE IN COLAB:
  1. Runtime → Change runtime type → T4 GPU → Save
  2. Upload kuala_lumpur_1min.csv when prompted
  3. Run all cells (Ctrl+F9)
  4. Download phase4_plots.zip at the end
=============================================================
"""

# ── Imports ───────────────────────────────────────────────────
import os, sys, shutil, time, math, warnings
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

# ── GPU / Device Setup ────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  Device: {device}")
if device.type == 'cuda':
    print(f"  GPU:    {torch.cuda.get_device_name(0)}")
    print(f"  VRAM:   {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ── Paths (Colab-safe, no __file__) ──────────────────────────
BASE_DIR     = '/content'
DATA_FILE    = '/content/kuala_lumpur_1min.csv'
RESULTS_FILE = '/content/phase4_results.txt'
PLOTS_DIR    = '/content/plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

def save_fig(filename):
    path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(path, bbox_inches='tight')
    print(f"  [SAVED] {path}")

# ── Upload CSV if not present ─────────────────────────────────
if not os.path.exists(DATA_FILE):
    print("kuala_lumpur_1min.csv not found. Uploading...")
    from google.colab import files
    uploaded = files.upload()
    # Move to /content if needed
    for fname in uploaded:
        if fname != 'kuala_lumpur_1min.csv':
            os.rename(fname, DATA_FILE)
    print("  Upload complete.")

# ── Plot style ────────────────────────────────────────────────
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
print("  PHASE 4: CLEARSKY-INDEX LSTM — KUALA LUMPUR SOLAR FORECASTING")
print("  Architecture: Conv1D + BiLSTM + Attention | k_t Normalization")
print("=" * 70)

# ─────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────
print("\n[1/7] Loading dataset...")
df = pd.read_csv(DATA_FILE)
df['DateTime'] = pd.to_datetime(df['DateTime'])
df['Date']     = df['DateTime'].dt.date
print(f"  Loaded: {len(df):,} rows | {df['Date'].nunique()} days")
print(f"  Columns: {df.columns.tolist()}")

# ─────────────────────────────────────────────────────────────
# 2. CLEAR-SKY GHI  (Ineichen-Perez / Kasten-Young approximation)
# ─────────────────────────────────────────────────────────────
print("\n[2/7] Computing Clear-Sky GHI and k_t index...")

LAT_DEG = 3.1390    # Kuala Lumpur
LON_DEG = 101.6869

def compute_clearsky_ghi(dt_series, lat_deg=LAT_DEG, lon_deg=LON_DEG):
    lat_rad = math.radians(lat_deg)
    doy  = dt_series.dt.dayofyear.values
    hour = (dt_series.dt.hour + dt_series.dt.minute / 60.0).values

    B    = 2 * np.pi * (doy - 1) / 365.0
    decl = (0.006918 - 0.399912*np.cos(B) + 0.070257*np.sin(B)
            - 0.006758*np.cos(2*B) + 0.000907*np.sin(2*B))
    EoT  = 229.18 * (0.000075 + 0.001868*np.cos(B)
           - 0.032077*np.sin(B) - 0.014615*np.cos(2*B) - 0.04089*np.sin(2*B))
    TC   = 4 * (lon_deg - 120.0) + EoT                 # UTC+8 meridian
    solar_time = hour + TC / 60.0
    H    = np.radians(15 * (solar_time - 12.0))

    cos_z = np.clip(
        np.sin(lat_rad)*np.sin(decl) + np.cos(lat_rad)*np.cos(decl)*np.cos(H),
        0.0, 1.0)

    E0 = 1 + 0.033 * np.cos(2 * np.pi * doy / 365.0)
    I0 = 1361.0 * E0

    z_deg = np.degrees(np.arccos(cos_z))
    AM    = np.where(
        cos_z > 0.01,
        1.0 / (cos_z + 0.50572 * np.power(np.clip(96.07995 - z_deg, 0.01, None), -1.6364)),
        0.0)

    # Ineichen-Perez (sea level, TL=2.7 tropical)
    TL = 2.7
    cg1 = 0.868; cg2 = 0.0387
    GHI_clear = I0 * cos_z * cg1 * np.exp(
        -cg2 * AM * (np.exp(0.01 * np.power(AM, 1.8))) * TL)
    return np.clip(GHI_clear, 0.0, None)

df['GHI_clear'] = compute_clearsky_ghi(df['DateTime'])
EPSILON = 5.0
df['kt'] = np.where(
    df['GHI_clear'] > EPSILON,
    np.clip(df['GHI'] / df['GHI_clear'], 0.0, 1.2),
    0.0)

print(f"  GHI_clear: {df['GHI_clear'].min():.1f} – {df['GHI_clear'].max():.1f} W/m²")
print(f"  k_t range: {df['kt'].min():.3f} – {df['kt'].max():.3f}")

# ─────────────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING (10 features)
# ─────────────────────────────────────────────────────────────
print("\n[3/7] Engineering 10-feature input set...")

doy_arr = df['DateTime'].dt.dayofyear.values
hour_arr = (df['DateTime'].dt.hour + df['DateTime'].dt.minute / 60.0).values
df['DOY_sin']    = np.sin(2 * np.pi * doy_arr  / 365.0)
df['DOY_cos']    = np.cos(2 * np.pi * doy_arr  / 365.0)
df['Hour_sin_p4']= np.sin(2 * np.pi * hour_arr / 24.0)
df['Hour_cos_p4']= np.cos(2 * np.pi * hour_arr / 24.0)

FEATURE_COLS = [
    'kt',               # 1. Clear-Sky Index
    'Temperature',      # 2. Ambient Temperature (°C)
    'Relative Humidity',# 3. Relative Humidity (%)
    'DNI',              # 4. Direct Normal Irradiance (W/m²)
    'DHI',              # 5. Diffuse Horizontal Irradiance (W/m²)
    'Pressure',         # 6. Atmospheric Pressure (hPa)
    'Hour_sin_p4',      # 7. Hour-of-day sine
    'Hour_cos_p4',      # 8. Hour-of-day cosine
    'DOY_sin',          # 9. Day-of-year sine
    'DOY_cos',          # 10. Day-of-year cosine
]
N_FEATURES = len(FEATURE_COLS)
LOOKBACK   = 60
HORIZONS   = [1, 5, 15, 30, 60]
print(f"  {N_FEATURES} features × {LOOKBACK} min = {N_FEATURES*LOOKBACK} data points/sample")

# ─────────────────────────────────────────────────────────────
# 4. TRAIN / VAL / TEST SPLIT + SCALING
# ─────────────────────────────────────────────────────────────
print("\n[4/7] Splitting and scaling...")
train_mask = df['Month'] <= 8
val_mask   = df['Month'] == 9
test_mask  = df['Month'] >= 10
print(f"  Train: {train_mask.sum():,} | Val: {val_mask.sum():,} | Test: {test_mask.sum():,}")

feature_scaler = StandardScaler()
feature_scaler.fit(df.loc[train_mask, FEATURE_COLS])
df_scaled = df.copy()
df_scaled[FEATURE_COLS] = feature_scaler.transform(df[FEATURE_COLS])

kt_idx     = FEATURE_COLS.index('kt')
kt_col_mean = feature_scaler.mean_[kt_idx]
kt_col_std  = feature_scaler.scale_[kt_idx]

# ─────────────────────────────────────────────────────────────
# 5. SEQUENCE EXTRACTION
# ─────────────────────────────────────────────────────────────
print("\n[5/7] Extracting sliding window sequences...")

feat_arr      = df_scaled[FEATURE_COLS].values.astype(np.float32)
kt_arr        = feat_arr[:, kt_idx]
ghi_clear_arr = df['GHI_clear'].values.astype(np.float32)
ghi_raw_arr   = df['GHI'].values.astype(np.float32)

def extract_sequences(dataframe, f_arr, kt_a, cl_arr, raw_arr, stride=1):
    Xs, Ykt, Ycl, Yact = [], [], [], []
    for d, grp in dataframe.groupby('Date'):
        idx = grp.index.values
        n   = len(idx)
        if n < LOOKBACK + 60:
            continue
        fe  = f_arr[idx]; kt = kt_a[idx]
        cl  = cl_arr[idx]; ra = raw_arr[idx]
        for t in range(LOOKBACK, n - 60 + 1, stride):
            Xs.append(fe[t-LOOKBACK:t])
            Ykt.append([kt[t], kt[t+4], kt[t+14], kt[t+29], kt[t+59]])
            Ycl.append([cl[t], cl[t+4], cl[t+14], cl[t+29], cl[t+59]])
            Yact.append([ra[t], ra[t+4], ra[t+14], ra[t+29], ra[t+59]])
    return (np.array(Xs, np.float32), np.array(Ykt, np.float32),
            np.array(Ycl, np.float32), np.array(Yact, np.float32))

X_tr, ykt_tr, ycl_tr, yact_tr = extract_sequences(df[train_mask], feat_arr, kt_arr, ghi_clear_arr, ghi_raw_arr, stride=3)
X_va, ykt_va, ycl_va, yact_va = extract_sequences(df[val_mask],   feat_arr, kt_arr, ghi_clear_arr, ghi_raw_arr, stride=2)
X_te, ykt_te, ycl_te, yact_te = extract_sequences(df[test_mask],  feat_arr, kt_arr, ghi_clear_arr, ghi_raw_arr, stride=1)
print(f"  Train: {X_tr.shape} | Val: {X_va.shape} | Test: {X_te.shape}")

BATCH = 512
def make_loader(X, y, shuffle):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=BATCH, shuffle=shuffle, pin_memory=(device.type=='cuda'), num_workers=2)

train_dl = make_loader(X_tr, ykt_tr, True)
val_dl   = make_loader(X_va, ykt_va, False)
test_dl  = make_loader(X_te, ykt_te, False)

# ─────────────────────────────────────────────────────────────
# 6. MODEL: Conv1D + BiLSTM + Attention
# ─────────────────────────────────────────────────────────────
class TemporalAttention(nn.Module):
    def __init__(self, h): super().__init__(); self.w = nn.Linear(h, 1, bias=False)
    def forward(self, x):  # x: (B,T,H)
        w = torch.softmax(self.w(x).squeeze(-1), dim=-1)   # (B,T)
        return (w.unsqueeze(-1) * x).sum(1), w              # (B,H)

class Phase4SolarModel(nn.Module):
    def __init__(self, n_feat=10, conv_ch=32, hidden=96, layers=2, n_out=5, drop=0.25):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_feat, conv_ch, 3, padding=1), nn.BatchNorm1d(conv_ch), nn.GELU(),
            nn.Conv1d(conv_ch, conv_ch, 3, padding=1), nn.BatchNorm1d(conv_ch), nn.GELU(),
        )
        self.lstm = nn.LSTM(conv_ch, hidden, layers, batch_first=True,
                            bidirectional=True, dropout=drop if layers>1 else 0.0)
        bi = hidden * 2
        self.attn = TemporalAttention(bi)
        self.head = nn.Sequential(
            nn.LayerNorm(bi), nn.Linear(bi, 64), nn.GELU(), nn.Dropout(drop),
            nn.Linear(64, 32), nn.GELU(), nn.Linear(32, n_out)
        )
    def forward(self, x):                      # x: (B,T,F)
        c = self.conv(x.permute(0,2,1))        # (B,conv_ch,T)
        c = c.permute(0,2,1)                   # (B,T,conv_ch)
        o, _ = self.lstm(c)                    # (B,T,2H)
        ctx, _ = self.attn(o)                  # (B,2H)
        return self.head(ctx)                  # (B,5)

model = Phase4SolarModel(n_feat=N_FEATURES).to(device)
params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nModel: Conv1D+BiLSTM+Attention | Params: {params:,} | Device: {device}")

# ─────────────────────────────────────────────────────────────
# 7. TRAINING
# ─────────────────────────────────────────────────────────────
print("\n[6/7] Training...")

# Weighted MSE: t+60 gets 3× weight vs t+1
HW = torch.tensor([1.0, 1.5, 2.0, 2.5, 3.0], device=device)

def weighted_mse(pred, tgt):
    return ((pred - tgt)**2 * HW).mean()

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
EPOCHS    = 40
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

best_val, best_state = float('inf'), None
train_hist, val_hist  = [], []

t0 = time.time()
for ep in range(1, EPOCHS+1):
    # Train
    model.train()
    tr = 0.0
    for bx, by in train_dl:
        bx, by = bx.to(device), by.to(device)
        optimizer.zero_grad()
        loss = weighted_mse(model(bx), by)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tr += loss.item() * bx.size(0)
    tr /= len(train_dl.dataset)
    train_hist.append(tr)

    # Validate
    model.eval()
    va = 0.0
    with torch.no_grad():
        for bx, by in val_dl:
            bx, by = bx.to(device), by.to(device)
            va += weighted_mse(model(bx), by).item() * bx.size(0)
    va /= len(val_dl.dataset)
    val_hist.append(va)
    scheduler.step()

    if va < best_val:
        best_val   = va
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if ep % 5 == 0 or ep == 1:
        print(f"  Ep {ep:3d}/{EPOCHS} | Train={tr:.5f} | Val={va:.5f} | "
              f"Best={best_val:.5f} | LR={optimizer.param_groups[0]['lr']:.1e}")

print(f"  Done in {time.time()-t0:.1f}s")
model.load_state_dict(best_state)

# ─────────────────────────────────────────────────────────────
# 8. EVALUATION
# ─────────────────────────────────────────────────────────────
print("\n[7/7] Evaluating on Oct–Dec test set...")
model.eval()
kt_preds = []
with torch.no_grad():
    for bx, _ in test_dl:
        kt_preds.append(model(bx.to(device)).cpu().numpy())
kt_pred_scaled = np.vstack(kt_preds)

# Unscale k_t
kt_pred   = kt_pred_scaled * kt_col_std + kt_col_mean
kt_actual = ykt_te         * kt_col_std + kt_col_mean

# Reconstruct GHI
ghi_pred   = np.clip(kt_pred   * ycl_te, 0, None)
ghi_actual = np.clip(yact_te,             0, None)

# Persistence baseline (last k_t in window × future GHI_clear)
kt_pers_s = X_te[:, -1, kt_idx]
kt_pers   = kt_pers_s * kt_col_std + kt_col_mean
ghi_pers  = np.clip(kt_pers[:, None] * ycl_te, 0, None)

p3_r2   = [0.8978, 0.7494, 0.6321, 0.5712, 0.5266]
p3_rmse = [91.22, 142.99, 173.90, 189.50, 204.67]

metrics = {}
print(f"\n  {'H':>4} {'R²(Ph3)':>9} {'R²(Ph4)':>9} {'ΔR²':>7}  "
      f"{'RMSE_Ph3':>10} {'RMSE_Ph4':>10}  {'Skill%':>8}")
print("  " + "─"*68)

for i, h in enumerate(HORIZONS):
    a = ghi_actual[:, i]; p = ghi_pred[:, i]; pe = ghi_pers[:, i]
    day = ycl_te[:, i] > EPSILON
    if day.sum() < 100: day = np.ones(len(a), bool)

    r2   = r2_score(a[day], p[day])
    mae  = mean_absolute_error(a[day], p[day])
    rmse = np.sqrt(mean_squared_error(a[day], p[day]))
    pr   = np.sqrt(mean_squared_error(a[day], pe[day]))
    ss   = (1 - rmse/pr) * 100
    metrics[h] = dict(R2=r2, MAE=mae, RMSE=rmse, P_RMSE=pr, Skill=ss)

    print(f"  +{h:<3d} {p3_r2[i]:>9.4f} {r2:>9.4f} {r2-p3_r2[i]:>+7.4f}  "
          f"{p3_rmse[i]:>10.2f} {rmse:>10.2f}  {ss:>+8.2f}%")

# ─────────────────────────────────────────────────────────────
# 9. PLOTS
# ─────────────────────────────────────────────────────────────
print("\nGenerating plots...")

# Plot 40: Training curve
fig, ax = plt.subplots(figsize=(9,5))
ax.plot(range(1,EPOCHS+1), train_hist, 'o-', color=CYAN, lw=2, ms=4, label='Train Loss')
ax.plot(range(1,EPOCHS+1), val_hist,   's--',color=ACCENT,lw=2, ms=4, label='Val Loss')
ax.set_xlabel('Epoch', fontweight='bold')
ax.set_ylabel('Weighted MSE (k_t space)', fontweight='bold')
ax.set_title('Phase 4 Training Convergence\nConv1D+BiLSTM+Attention | CosineAnnealing LR', fontweight='bold')
ax.legend(); plt.tight_layout(); save_fig('40_ph4_training_convergence.png'); plt.close()

# Plot 41: R² & RMSE comparison Phase 3 vs Phase 4
fig, axes = plt.subplots(1, 2, figsize=(14,5))
x = np.arange(len(HORIZONS)); w = 0.35
hl = [f'+{h}m' for h in HORIZONS]
r2_ph4   = [metrics[h]['R2']   for h in HORIZONS]
rmse_ph4 = [metrics[h]['RMSE'] for h in HORIZONS]

b3 = axes[0].bar(x-w/2, p3_r2,  w, label='Phase 3 (Raw GHI LSTM)', color=CYAN,   edgecolor=BLUE, alpha=0.8)
b4 = axes[0].bar(x+w/2, r2_ph4, w, label='Phase 4 (k_t BiLSTM)',   color=ACCENT, edgecolor='#8B3A0C', alpha=0.9)
for b in b3: axes[0].text(b.get_x()+b.get_width()/2, b.get_height()+0.01, f'{b.get_height():.3f}', ha='center', fontsize=8)
for b in b4: axes[0].text(b.get_x()+b.get_width()/2, b.get_height()+0.01, f'{b.get_height():.3f}', ha='center', fontsize=8, fontweight='bold')
axes[0].set_xticks(x); axes[0].set_xticklabels(hl)
axes[0].set_ylabel('R² Score', fontweight='bold')
axes[0].set_title('Phase 3 vs Phase 4 — R² Comparison', fontweight='bold')
axes[0].legend(); axes[0].set_ylim(0, 1.1)
axes[0].axhline(0.8, color='green', ls=':', lw=1.5, label='R²=0.80 target')

axes[1].plot(HORIZONS, p3_rmse,  'o-', color=CYAN,  lw=2.5, ms=7, label='Phase 3 RMSE')
axes[1].plot(HORIZONS, rmse_ph4, 's-', color=ACCENT, lw=2.5, ms=7, label='Phase 4 RMSE')
axes[1].fill_between(HORIZONS, p3_rmse, rmse_ph4,
    where=[r4<r3 for r3,r4 in zip(p3_rmse,rmse_ph4)], color='green', alpha=0.2, label='Improvement')
axes[1].fill_between(HORIZONS, p3_rmse, rmse_ph4,
    where=[r4>=r3 for r3,r4 in zip(p3_rmse,rmse_ph4)], color='red', alpha=0.2, label='Regression')
axes[1].set_xlabel('Horizon (min)', fontweight='bold'); axes[1].set_ylabel('RMSE (W/m²)', fontweight='bold')
axes[1].set_title('Phase 3 vs Phase 4 — RMSE', fontweight='bold')
axes[1].set_xticks(HORIZONS); axes[1].legend()
plt.tight_layout(); save_fig('41_ph4_vs_ph3_comparison.png'); plt.close()

# Plot 42: +60 min scatter & residuals
act60 = ghi_actual[:,4]; pred60 = ghi_pred[:,4]
day60 = ycl_te[:,4] > EPSILON
sid   = np.random.choice(np.where(day60)[0], size=min(8000,day60.sum()), replace=False)

fig, axes = plt.subplots(1,2,figsize=(14,6))
axes[0].scatter(act60[sid], pred60[sid], alpha=0.3, s=10, color=ACCENT, edgecolors='none')
axes[0].plot([0,1200],[0,1200],'k--',lw=2,label='1:1 Ideal')
axes[0].set_xlabel('Actual GHI (W/m²)', fontweight='bold')
axes[0].set_ylabel('Predicted GHI (W/m²)', fontweight='bold')
axes[0].set_title(f'Phase 4 — +60 Min GHI Scatter\nR²={metrics[60]["R2"]:.4f} | RMSE={metrics[60]["RMSE"]:.2f} W/m²', fontweight='bold')
axes[0].legend(); axes[0].set_xlim(0,1200); axes[0].set_ylim(0,1200)

res60 = pred60[sid]-act60[sid]
sns.histplot(res60, bins=60, kde=True, color=ACCENT, ax=axes[1])
axes[1].axvline(0,color='k',ls='--',lw=1.5)
axes[1].set_xlabel('Error (W/m²)', fontweight='bold')
axes[1].set_title(f'+60 Min Error Distribution\nBias={np.mean(res60):.2f} | Std={np.std(res60):.2f} W/m²', fontweight='bold')
plt.tight_layout(); save_fig('42_ph4_60min_scatter_residuals.png'); plt.close()

# Plot 43: Walk-forward (+1 and +60)
fig, axes = plt.subplots(2,1,figsize=(14,9))
n_show = min(400, ghi_actual.shape[0])
t_arr  = np.arange(n_show)
for ax_i, (h_idx, h) in enumerate([(0,1),(4,60)]):
    a = ghi_actual[:n_show, h_idx]; p = ghi_pred[:n_show, h_idx]; pe = ghi_pers[:n_show, h_idx]
    axes[ax_i].plot(t_arr, a,  color=ACCENT, lw=2,   label='Actual GHI')
    axes[ax_i].plot(t_arr, p,  color=CYAN,   lw=2,   ls='--', label=f'Phase 4 (+{h}m)')
    axes[ax_i].plot(t_arr, pe, color='gray', lw=1.2, ls=':',  alpha=0.7, label='Persistence')
    axes[ax_i].fill_between(t_arr, a, p, color='lightblue', alpha=0.25)
    axes[ax_i].set_title(f'+{h} Min | R²={r2_score(a,p):.4f} | MAE={mean_absolute_error(a,p):.1f} W/m²', fontweight='bold')
    axes[ax_i].set_ylabel('GHI (W/m²)', fontweight='bold'); axes[ax_i].legend()
axes[1].set_xlabel('Sample Index', fontweight='bold')
plt.suptitle('Phase 4 Walk-Forward Forecast — Conv1D+BiLSTM+Attention', fontweight='bold', y=1.01)
plt.tight_layout(); save_fig('43_ph4_walkforward_t1_t60.png'); plt.close()

# Plot 44: Skill score comparison
sk_ph3 = [3.07, 11.66, 17.16, 20.30, 25.79]
sk_ph4 = [metrics[h]['Skill'] for h in HORIZONS]
fig, ax = plt.subplots(figsize=(9,5))
ax.plot(HORIZONS, sk_ph4, 's-',  color=ACCENT, lw=2.5, ms=8, label='Phase 4 Skill')
ax.plot(HORIZONS, sk_ph3, 'o--', color=CYAN,   lw=2.0, ms=7, label='Phase 3 Skill')
ax.fill_between(HORIZONS, sk_ph3, sk_ph4,
    where=[s4>s3 for s3,s4 in zip(sk_ph3,sk_ph4)], color='green', alpha=0.2, label='Phase 4 gain')
ax.fill_between(HORIZONS, sk_ph3, sk_ph4,
    where=[s4<=s3 for s3,s4 in zip(sk_ph3,sk_ph4)], color='red', alpha=0.2, label='Phase 3 better')
ax.set_xlabel('Horizon (min)', fontweight='bold')
ax.set_ylabel('IEEE Skill Score over Persistence (%)', fontweight='bold')
ax.set_title('Phase 4 vs Phase 3 — Forecast Skill Score', fontweight='bold')
ax.set_xticks(HORIZONS); ax.legend()
plt.tight_layout(); save_fig('44_ph4_skill_score_comparison.png'); plt.close()

# ─────────────────────────────────────────────────────────────
# 10. SAVE RESULTS + DOWNLOAD
# ─────────────────────────────────────────────────────────────
with open(RESULTS_FILE, 'w') as f:
    f.write("PHASE 4: CLEAR-SKY INDEX LSTM RESULTS\n")
    f.write("=" * 70 + "\n\n")
    f.write("APPROACH: k_t = GHI/GHI_clear | Conv1D+BiLSTM+Attention\n")
    f.write(f"Params: {params:,} | Device: {device}\n\n")
    f.write(f"{'H':<5} {'R²(Ph3)':<10} {'R²(Ph4)':<10} {'ΔR²':<8} "
            f"{'RMSE_Ph3':<10} {'RMSE_Ph4':<10} {'MAE':<10} {'Skill%'}\n")
    f.write("─"*72+"\n")
    for i, h in enumerate(HORIZONS):
        m = metrics[h]
        f.write(f"+{h:<4d} {p3_r2[i]:<10.4f} {m['R2']:<10.4f} {m['R2']-p3_r2[i]:<+8.4f} "
                f"{p3_rmse[i]:<10.2f} {m['RMSE']:<10.2f} {m['MAE']:<10.2f} {m['Skill']:+.2f}%\n")

print(f"\nResults saved to {RESULTS_FILE}")

# Zip everything and download
shutil.make_archive('/content/phase4_output', 'zip', '/content/plots')
import shutil as _sh
_sh.copy2(RESULTS_FILE, '/content/plots/phase4_results.txt')
shutil.make_archive('/content/phase4_output', 'zip', '/content/plots')

print("\nDownloading results...")
from google.colab import files
files.download('/content/phase4_results.txt')
files.download('/content/phase4_output.zip')

print("\n" + "="*70)
print("  [DONE] Phase 4 Complete!")
print("="*70)

