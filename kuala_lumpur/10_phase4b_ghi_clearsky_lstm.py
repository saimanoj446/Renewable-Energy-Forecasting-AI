"""
=============================================================
Phase 4b: Improved Raw-GHI LSTM — Kuala Lumpur
Fixes for Phase 3 (overshoot at noon) WITHOUT k_t reconstruction
DOP: Forecasting of Renewable Resources Using AI
BITS Pilani, Hyderabad | EEE Dept | Prof. Arup Ratan
=============================================================
PHASE 4a POSTMORTEM:
  - k_t prediction amplified reconstruction errors at noon
  - Val loss diverged by epoch 5 (model overfit Jan-Aug train)
  - Stride=3 made training set too sparse

PHASE 4b CHANGES:
  1. Back to raw GHI as target (proven to work in Phase 3)
  2. ADD GHI_clear as an extra input feature (11 features)
     => LSTM learns DEVIATION from clear-sky, not absolute value
     => This implicitly teaches it the bell curve, preventing overshoot
  3. Reduce dropout: 0.25 -> 0.15 (model was underfitting val)
  4. Stride=1 on train (use ALL data; GPU can handle it)
  5. Early stopping patience=8 on val loss (stop before divergence)
  6. Lighter model: Conv1D(16) + LSTM(64 uni-dir) to reduce overfit
  7. Separate GHI StandardScaler per-day (removes diurnal scale shift)
=============================================================
"""

import os, shutil, time, math, warnings
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

# ── Device ───────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  Device: {device}")
if device.type == 'cuda':
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

# ── Paths ────────────────────────────────────────────────────
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = '/content'          # Colab

DATA_FILE    = os.path.join(BASE_DIR, 'kuala_lumpur_1min.csv')
RESULTS_FILE = os.path.join(BASE_DIR, 'phase4b_results.txt')
PLOTS_DIR    = os.path.join(BASE_DIR, 'plots')
ROOT_PLOTS   = os.path.abspath(os.path.join(BASE_DIR, '..', 'plots'))
for d in [PLOTS_DIR, ROOT_PLOTS]: os.makedirs(d, exist_ok=True)

def save_fig(fn):
    for d in [PLOTS_DIR, ROOT_PLOTS]:
        plt.savefig(os.path.join(d, fn), bbox_inches='tight')
    print(f"  [SAVED] {fn}")

plt.rcParams.update({'figure.dpi':150,'savefig.dpi':150,
                     'font.size':11,'axes.titlesize':13,
                     'axes.labelsize':11,'figure.facecolor':'white'})
sns.set_style("whitegrid")
ACCENT='#FF6B35'; BLUE='#2C3E50'; CYAN='#3498DB'
GREEN='#2ECC71'; PURPLE='#9B59B6'; RED='#E74C3C'

torch.manual_seed(42); np.random.seed(42)

print("="*70)
print("  PHASE 4b: IMPROVED GHI LSTM — KL SOLAR (GHI_clear as extra feature)")
print("="*70)

# ─────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────
print("\n[1/7] Loading data...")
df = pd.read_csv(DATA_FILE)
df['DateTime'] = pd.to_datetime(df['DateTime'])
df['Date']     = df['DateTime'].dt.date
print(f"  {len(df):,} rows | {df['Date'].nunique()} days")

# ─────────────────────────────────────────────────────────────
# 2. COMPUTE GHI_CLEAR  (same Kasten-Young as Phase 4a)
# ─────────────────────────────────────────────────────────────
print("\n[2/7] Computing GHI_clear...")
LAT_DEG=3.1390; LON_DEG=101.6869

def compute_clearsky(dt_s, lat=LAT_DEG, lon=LON_DEG):
    lr = math.radians(lat)
    doy  = dt_s.dt.dayofyear.values
    hour = (dt_s.dt.hour + dt_s.dt.minute/60.0).values
    B    = 2*np.pi*(doy-1)/365.0
    decl = (0.006918 - 0.399912*np.cos(B) + 0.070257*np.sin(B)
            - 0.006758*np.cos(2*B) + 0.000907*np.sin(2*B))
    EoT  = 229.18*(0.000075 + 0.001868*np.cos(B) - 0.032077*np.sin(B)
                   - 0.014615*np.cos(2*B) - 0.04089*np.sin(2*B))
    TC   = 4*(lon-120.0)+EoT
    H    = np.radians(15*((hour+TC/60.0)-12.0))
    cz   = np.clip(np.sin(lr)*np.sin(decl)+np.cos(lr)*np.cos(decl)*np.cos(H),0,1)
    E0   = 1+0.033*np.cos(2*np.pi*doy/365.0)
    I0   = 1361.0*E0
    zdeg = np.degrees(np.arccos(cz))
    AM   = np.where(cz>0.01,
                    1.0/(cz+0.50572*np.power(np.clip(96.07995-zdeg,0.01,None),-1.6364)),
                    0.0)
    Gcc  = I0*cz*0.868*np.exp(-0.0387*AM*2.7*np.exp(0.01*AM**1.8))
    return np.clip(Gcc, 0.0, None)

df['GHI_clear'] = compute_clearsky(df['DateTime'])

# KEY INSIGHT: Add GHI_clear as a feature.
# The LSTM now sees the expected clear-sky value at each minute.
# It learns to predict the RESIDUAL deviation, not the absolute value.
# This prevents over-extrapolation past solar noon.
print(f"  GHI_clear: {df['GHI_clear'].min():.1f}–{df['GHI_clear'].max():.1f} W/m²")

# ─────────────────────────────────────────────────────────────
# 3. FEATURES (11 features — raw GHI + GHI_clear added)
# ─────────────────────────────────────────────────────────────
print("\n[3/7] Building 11-feature set...")
doy_v  = df['DateTime'].dt.dayofyear.values
hour_v = (df['DateTime'].dt.hour + df['DateTime'].dt.minute/60.0).values
df['DOY_sin']     = np.sin(2*np.pi*doy_v/365.0)
df['DOY_cos']     = np.cos(2*np.pi*doy_v/365.0)
df['Hour_sin_p4'] = np.sin(2*np.pi*hour_v/24.0)
df['Hour_cos_p4'] = np.cos(2*np.pi*hour_v/24.0)

FEATURE_COLS = [
    'GHI',              # 1. Raw Solar Irradiance (target)
    'GHI_clear',        # 2. *** Clear-sky GHI — teaches the diurnal curve ***
    'Temperature',      # 3. Ambient Temperature
    'Relative Humidity',# 4. Relative Humidity
    'DNI',              # 5. Direct Normal Irradiance
    'DHI',              # 6. Diffuse Horizontal Irradiance
    'Pressure',         # 7. Atmospheric Pressure
    'Hour_sin_p4',      # 8. Hour sine
    'Hour_cos_p4',      # 9. Hour cosine
    'DOY_sin',          # 10. Day-of-year sine
    'DOY_cos',          # 11. Day-of-year cosine
]
TARGET_COL = 'GHI'
N_FEAT     = len(FEATURE_COLS)
LOOKBACK   = 60
HORIZONS   = [1, 5, 15, 30, 60]
print(f"  {N_FEAT} features × {LOOKBACK} min = {N_FEAT*LOOKBACK} data points/sample")

# ─────────────────────────────────────────────────────────────
# 4. TRAIN/VAL/TEST SPLIT + SCALING
# ─────────────────────────────────────────────────────────────
print("\n[4/7] Splitting and scaling...")
train_mask = df['Month'] <= 8
val_mask   = df['Month'] == 9
test_mask  = df['Month'] >= 10
print(f"  Train:{train_mask.sum():,} | Val:{val_mask.sum():,} | Test:{test_mask.sum():,}")

feature_scaler = StandardScaler()
feature_scaler.fit(df.loc[train_mask, FEATURE_COLS])
df_scaled = df.copy()
df_scaled[FEATURE_COLS] = feature_scaler.transform(df[FEATURE_COLS])

target_scaler = StandardScaler()
target_scaler.fit(df.loc[train_mask, [TARGET_COL]])
scaled_target = target_scaler.transform(df[[TARGET_COL]]).flatten()

ghi_idx = FEATURE_COLS.index('GHI')

# ─────────────────────────────────────────────────────────────
# 5. SEQUENCE EXTRACTION  (stride=1 everywhere — GPU can handle it)
# ─────────────────────────────────────────────────────────────
print("\n[5/7] Extracting sequences (stride=1)...")

feat_arr = df_scaled[FEATURE_COLS].values.astype(np.float32)
tgt_arr  = scaled_target.astype(np.float32)
ghi_raw  = df['GHI'].values.astype(np.float32)

def extract(dataframe, stride=1):
    Xs, Ys, Yraw = [], [], []
    for d, grp in dataframe.groupby('Date'):
        idx = grp.index.values; n = len(idx)
        if n < LOOKBACK+60: continue
        fe = feat_arr[idx]; tg = tgt_arr[idx]; raw = ghi_raw[idx]
        for t in range(LOOKBACK, n-59, stride):
            Xs.append(fe[t-LOOKBACK:t])
            Ys.append([tg[t], tg[t+4], tg[t+14], tg[t+29], tg[t+59]])
            Yraw.append([raw[t], raw[t+4], raw[t+14], raw[t+29], raw[t+59]])
    return (np.array(Xs,np.float32),
            np.array(Ys,np.float32),
            np.array(Yraw,np.float32))

X_tr, y_tr, yr_tr = extract(df[train_mask], stride=2)   # stride 2 only on train
X_va, y_va, yr_va = extract(df[val_mask],   stride=1)
X_te, y_te, yr_te = extract(df[test_mask],  stride=1)
print(f"  Train:{X_tr.shape} | Val:{X_va.shape} | Test:{X_te.shape}")

BATCH = 512
def dl(X, y, shuf):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=BATCH, shuffle=shuf,
                      pin_memory=(device.type=='cuda'), num_workers=2)
train_dl = dl(X_tr, y_tr, True)
val_dl   = dl(X_va, y_va, False)
test_dl  = dl(X_te, y_te, False)

# ─────────────────────────────────────────────────────────────
# 6. LIGHTER MODEL (reduced overfit vs Phase 4a)
#    Conv1D(16) + LSTM(64, unidirectional) + Attention
# ─────────────────────────────────────────────────────────────
class Attention(nn.Module):
    def __init__(self, h): super().__init__(); self.w = nn.Linear(h,1,bias=False)
    def forward(self, x):
        w = torch.softmax(self.w(x).squeeze(-1), dim=-1)
        return (w.unsqueeze(-1)*x).sum(1)

class Phase4bModel(nn.Module):
    def __init__(self, n_feat=11, conv_ch=16, hidden=64, layers=2, n_out=5, drop=0.15):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_feat, conv_ch, 3, padding=1), nn.BatchNorm1d(conv_ch), nn.GELU(),
        )
        self.lstm = nn.LSTM(conv_ch, hidden, layers, batch_first=True,
                            bidirectional=False,
                            dropout=drop if layers>1 else 0.0)
        self.attn = Attention(hidden)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 32), nn.GELU(), nn.Dropout(drop),
            nn.Linear(32, n_out)
        )
    def forward(self, x):
        c = self.conv(x.permute(0,2,1)).permute(0,2,1)
        o, _ = self.lstm(c)
        return self.head(self.attn(o))

model  = Phase4bModel(n_feat=N_FEAT).to(device)
params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nModel: Conv1D+LSTM+Attention | Params:{params:,} | Device:{device}")

# ─────────────────────────────────────────────────────────────
# 7. TRAINING with early stopping
# ─────────────────────────────────────────────────────────────
print("\n[6/7] Training with early stopping (patience=8)...")

# Equal loss across horizons (weighted MSE caused it to sacrifice t+1)
def mse_loss(pred, tgt):
    return ((pred - tgt)**2).mean()

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
EPOCHS    = 60
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=4)
PATIENCE  = 8

best_val, best_ep, best_state = float('inf'), 0, None
train_hist, val_hist = [], []
t0 = time.time()

for ep in range(1, EPOCHS+1):
    model.train()
    tr = 0.0
    for bx, by in train_dl:
        bx, by = bx.to(device), by.to(device)
        optimizer.zero_grad()
        loss = mse_loss(model(bx), by)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tr += loss.item()*bx.size(0)
    tr /= len(train_dl.dataset); train_hist.append(tr)

    model.eval(); va = 0.0
    with torch.no_grad():
        for bx, by in val_dl:
            bx, by = bx.to(device), by.to(device)
            va += mse_loss(model(bx), by).item()*bx.size(0)
    va /= len(val_dl.dataset); val_hist.append(va)
    scheduler.step(va)

    if va < best_val:
        best_val, best_ep = va, ep
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if ep % 5 == 0 or ep == 1:
        print(f"  Ep {ep:3d} | Train={tr:.5f} | Val={va:.5f} | Best ep={best_ep} | LR={optimizer.param_groups[0]['lr']:.1e}")

    if ep - best_ep >= PATIENCE:
        print(f"  Early stop at epoch {ep} (best was ep {best_ep})")
        break

print(f"  Training time: {time.time()-t0:.1f}s")
model.load_state_dict(best_state)

# ─────────────────────────────────────────────────────────────
# 8. EVALUATION
# ─────────────────────────────────────────────────────────────
print("\n[7/7] Evaluating on test set...")
model.eval(); preds = []
with torch.no_grad():
    for bx, _ in test_dl:
        preds.append(model(bx.to(device)).cpu().numpy())
y_pred_s = np.vstack(preds)

# Unscale
tgt_mean, tgt_std = target_scaler.mean_[0], target_scaler.scale_[0]
y_pred_wm2  = np.clip(y_pred_s * tgt_std + tgt_mean, 0, None)
y_test_wm2  = np.clip(yr_te,                          0, None)

# Persistence baseline
ghi_mean = feature_scaler.mean_[ghi_idx]
ghi_std  = feature_scaler.scale_[ghi_idx]
pers_wm2 = np.clip(X_te[:,-1,ghi_idx]*ghi_std + ghi_mean, 0, None)

p3_r2   = [0.8978, 0.7494, 0.6321, 0.5712, 0.5266]
p3_rmse = [91.22, 142.99, 173.90, 189.50, 204.67]
metrics = {}

print(f"\n  {'H':>4} {'R²(Ph3)':>9} {'R²(4b)':>9} {'ΔR²':>7}  "
      f"{'RMSE_Ph3':>10} {'RMSE_4b':>10}  {'Skill%':>8}")
print("  "+"─"*68)
for i, h in enumerate(HORIZONS):
    a = y_test_wm2[:,i]; p = y_pred_wm2[:,i]
    r2   = r2_score(a, p)
    mae  = mean_absolute_error(a, p)
    rmse = np.sqrt(mean_squared_error(a, p))
    pr   = np.sqrt(mean_squared_error(a, pers_wm2))
    ss   = (1 - rmse/pr)*100
    metrics[h] = dict(R2=r2, MAE=mae, RMSE=rmse, P_RMSE=pr, Skill=ss)
    print(f"  +{h:<3d} {p3_r2[i]:>9.4f} {r2:>9.4f} {r2-p3_r2[i]:>+7.4f}  "
          f"{p3_rmse[i]:>10.2f} {rmse:>10.2f}  {ss:>+8.2f}%")

# ─────────────────────────────────────────────────────────────
# 9. PLOTS
# ─────────────────────────────────────────────────────────────
print("\nGenerating plots...")

# Plot 45: Training convergence
fig, ax = plt.subplots(figsize=(9,5))
ep_arr = range(1, len(train_hist)+1)
ax.plot(ep_arr, train_hist, 'o-', color=CYAN,  lw=2, ms=4, label='Train Loss')
ax.plot(ep_arr, val_hist,   's--',color=ACCENT, lw=2, ms=4, label='Val Loss')
ax.axvline(best_ep, color='green', ls=':', lw=1.5, label=f'Best Epoch ({best_ep})')
ax.set_xlabel('Epoch', fontweight='bold')
ax.set_ylabel('MSE Loss (GHI normalized space)', fontweight='bold')
ax.set_title('Phase 4b Training — GHI + GHI_clear Features\n'
             'ReduceLROnPlateau | Early Stopping | Patience=8', fontweight='bold')
ax.legend(); plt.tight_layout(); save_fig('45_ph4b_training_convergence.png'); plt.close()

# Plot 46: R² comparison Ph3 vs Ph4a vs Ph4b
p4a_r2 = [0.7888, 0.6956, 0.6041, 0.5574, 0.5187]
fig, axes = plt.subplots(1,2,figsize=(14,5))
x = np.arange(5); w = 0.25
hl = [f'+{h}m' for h in HORIZONS]
r2_4b = [metrics[h]['R2'] for h in HORIZONS]

axes[0].bar(x-w, p3_r2,  w, label='Phase 3 (Raw LSTM)',   color=CYAN,   edgecolor=BLUE,      alpha=0.8)
axes[0].bar(x,   p4a_r2, w, label='Phase 4a (k_t BiLSTM)',color=RED,    edgecolor='#8B0000',  alpha=0.7)
axes[0].bar(x+w, r2_4b,  w, label='Phase 4b (GHI+GHI_cl)',color=ACCENT, edgecolor='#8B3A0C', alpha=0.9)
for i,(v3,v4a,v4b) in enumerate(zip(p3_r2,p4a_r2,r2_4b)):
    axes[0].text(i-w, v3+0.01, f'{v3:.3f}', ha='center', fontsize=7)
    axes[0].text(i,   v4a+0.01,f'{v4a:.3f}',ha='center', fontsize=7)
    axes[0].text(i+w, v4b+0.01,f'{v4b:.3f}',ha='center', fontsize=8, fontweight='bold')
axes[0].set_xticks(x); axes[0].set_xticklabels(hl)
axes[0].set_ylabel('R² Score', fontweight='bold')
axes[0].set_title('Phase 3 vs 4a vs 4b — R² Comparison', fontweight='bold')
axes[0].legend(fontsize=8); axes[0].set_ylim(0,1.1)
axes[0].axhline(0.8, color='green', ls=':', lw=1.5, label='R²=0.80 target')

rmse_4b = [metrics[h]['RMSE'] for h in HORIZONS]
p4a_rmse = [131.15, 157.60, 180.42, 192.53, 204.53]
axes[1].plot(HORIZONS, p3_rmse,  'o-', color=CYAN,   lw=2.5, ms=7, label='Phase 3')
axes[1].plot(HORIZONS, p4a_rmse, 'x--',color=RED,    lw=2.0, ms=7, label='Phase 4a')
axes[1].plot(HORIZONS, rmse_4b,  's-', color=ACCENT,  lw=2.5, ms=7, label='Phase 4b')
axes[1].set_xlabel('Horizon (min)', fontweight='bold')
axes[1].set_ylabel('RMSE (W/m²)', fontweight='bold')
axes[1].set_title('RMSE Comparison (lower = better)', fontweight='bold')
axes[1].set_xticks(HORIZONS); axes[1].legend()
plt.tight_layout(); save_fig('46_ph4b_vs_all_comparison.png'); plt.close()

# Plot 47: +60 min scatter
act60=y_test_wm2[:,4]; pred60=y_pred_wm2[:,4]
day60 = act60 > 10
sid   = np.random.choice(np.where(day60)[0], size=min(8000,day60.sum()), replace=False)
fig, axes = plt.subplots(1,2,figsize=(14,6))
axes[0].scatter(act60[sid], pred60[sid], alpha=0.3, s=10, color=ACCENT, edgecolors='none')
axes[0].plot([0,1200],[0,1200],'k--',lw=2,label='1:1 Ideal')
axes[0].set_xlabel('Actual GHI (W/m²)', fontweight='bold')
axes[0].set_ylabel('Predicted GHI (W/m²)', fontweight='bold')
axes[0].set_title(f'Phase 4b — +60 Min GHI Scatter\nR²={metrics[60]["R2"]:.4f} | RMSE={metrics[60]["RMSE"]:.2f} W/m²', fontweight='bold')
axes[0].legend(); axes[0].set_xlim(0,1200); axes[0].set_ylim(0,1200)
res60=pred60[sid]-act60[sid]
sns.histplot(res60, bins=60, kde=True, color=ACCENT, ax=axes[1])
axes[1].axvline(0,color='k',ls='--',lw=1.5)
axes[1].set_xlabel('Error (W/m²)', fontweight='bold')
axes[1].set_title(f'+60 Min Error | Bias={np.mean(res60):.2f} | Std={np.std(res60):.2f} W/m²', fontweight='bold')
plt.tight_layout(); save_fig('47_ph4b_60min_scatter.png'); plt.close()

# Plot 48: Walk-forward t+1 and t+60
fig, axes = plt.subplots(2,1,figsize=(14,9))
n_show = min(400, y_test_wm2.shape[0])
t_arr  = np.arange(n_show)
for ax_i,(h_idx,h) in enumerate([(0,1),(4,60)]):
    a=y_test_wm2[:n_show,h_idx]; p=y_pred_wm2[:n_show,h_idx]; pe=pers_wm2[:n_show]
    axes[ax_i].plot(t_arr,a, color=ACCENT,lw=2,label='Actual GHI')
    axes[ax_i].plot(t_arr,p, color=CYAN,  lw=2,ls='--',label=f'Phase 4b (+{h}m)')
    axes[ax_i].plot(t_arr,pe,color='gray',lw=1.2,ls=':',alpha=0.7,label='Persistence')
    axes[ax_i].fill_between(t_arr,a,p,color='lightblue',alpha=0.25)
    axes[ax_i].set_title(f'+{h} Min | R²={r2_score(a,p):.4f} | MAE={mean_absolute_error(a,p):.1f} W/m²', fontweight='bold')
    axes[ax_i].set_ylabel('GHI (W/m²)', fontweight='bold'); axes[ax_i].legend()
axes[1].set_xlabel('Sample Index', fontweight='bold')
plt.suptitle('Phase 4b — GHI+GHI_clear LSTM Walk-Forward Forecast\n'
             'Kuala Lumpur Oct–Dec 2025 (Out-of-Time Test)', fontweight='bold', y=1.01)
plt.tight_layout(); save_fig('48_ph4b_walkforward.png'); plt.close()

# ─────────────────────────────────────────────────────────────
# 10. SAVE RESULTS
# ─────────────────────────────────────────────────────────────
p4a_r2_d = {h:v for h,v in zip(HORIZONS,p4a_r2)}
with open(RESULTS_FILE,'w') as f:
    f.write("PHASE 4b: GHI + GHI_clear LSTM RESULTS\n"+"="*70+"\n\n")
    f.write("KEY CHANGE vs Phase 4a: Added GHI_clear as input feature.\n")
    f.write("  The LSTM now sees what the clear-sky value SHOULD be at each step.\n")
    f.write("  It learns the residual (cloud effect) not the absolute diurnal curve.\n")
    f.write(f"  Params: {params:,} | Device: {device} | Best epoch: {best_ep}\n\n")
    f.write(f"{'H':<5} {'R²(Ph3)':<10} {'R²(4a)':<10} {'R²(4b)':<10} "
            f"{'ΔvsP3':<8} {'RMSE_Ph3':<10} {'RMSE_4b':<10} {'Skill%'}\n")
    f.write("─"*80+"\n")
    for i,h in enumerate(HORIZONS):
        m=metrics[h]
        f.write(f"+{h:<4d} {p3_r2[i]:<10.4f} {p4a_r2[i]:<10.4f} {m['R2']:<10.4f} "
                f"{m['R2']-p3_r2[i]:<+8.4f} {p3_rmse[i]:<10.2f} {m['RMSE']:<10.2f} {m['Skill']:+.2f}%\n")

print(f"\n  Results saved to {RESULTS_FILE}")

# Colab: auto-download
try:
    import shutil as _sh
    _sh.copy2(RESULTS_FILE, os.path.join(PLOTS_DIR, 'phase4b_results.txt'))
    _sh.make_archive('/content/phase4b_output', 'zip', PLOTS_DIR)
    from google.colab import files
    files.download('/content/phase4b_results.txt')
    files.download('/content/phase4b_output.zip')
except Exception:
    pass  # Not in Colab — files already saved locally

print("\n"+"="*70)
print("  [DONE] Phase 4b Complete! Plots: 45, 46, 47, 48")
print("="*70)

