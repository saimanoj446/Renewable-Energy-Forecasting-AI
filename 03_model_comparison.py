"""
=============================================================
Phase 2: Model Comparison Study
DOP: Forecasting of Renewable Resources Using AI
BITS Pilani, Hyderabad | EEE Dept | Prof. Arup Ratan
=============================================================
Models Compared:
  1. Random Forest Regressor       (Phase 1 baseline)
  2. Gradient Boosting Regressor
  3. XGBoost Regressor
  4. Support Vector Regressor (SVR)

Improvements over Phase 1:
  - Lag features (previous 1h, 2h, 3h GHI)
  - Rolling averages (3h, 6h window)
  - Time-series cross-validation (TimeSeriesSplit)
  - Hyperparameter tuning via RandomizedSearchCV
  - Full model benchmarking with comparison plots
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import time
import os
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[WARNING] XGBoost not installed. Run: pip install xgboost")
    print("         Continuing without XGBoost...\n")

# ── Plot Style ────────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.facecolor': 'white',
})
sns.set_style("whitegrid")
os.makedirs('plots', exist_ok=True)

# ── Color Palette (consistent across all plots) ────────────────
MODEL_COLORS = {
    'Random Forest':      '#4A90D9',
    'Gradient Boosting':  '#E74C3C',
    'XGBoost':            '#2ECC71',
    'SVR':                '#9B59B6',
}

# =============================================================
# 1. Load & Engineer Features
# =============================================================
print("=" * 65)
print("  PHASE 2: MODEL COMPARISON — SOLAR IRRADIANCE FORECASTING")
print("=" * 65)
print("\nLoading data from BA_Combined.xlsx ...")
df = pd.read_excel('BA_Combined.xlsx', sheet_name='Sheet1')

# Datetime index
df['DateTime'] = pd.to_datetime(
    df[['Year', 'Month', 'Day', 'Hour']].assign(Minute=30)
)
df = df.sort_values('DateTime').reset_index(drop=True)

# ── Basic time features ───────────────────────────────────────
df['DayOfYear'] = df['DateTime'].dt.dayofyear
df['Hour_sin']  = np.sin(2 * np.pi * df['Hour'] / 24)
df['Hour_cos']  = np.cos(2 * np.pi * df['Hour'] / 24)
df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)

# ── Phase 2 additions: Lag features & Rolling averages ───────
print("Engineering lag features and rolling averages (Phase 2) ...")
df['GHI_lag1']       = df['GHI'].shift(1)
df['GHI_lag2']       = df['GHI'].shift(2)
df['GHI_lag3']       = df['GHI'].shift(3)
df['GHI_roll3']      = df['GHI'].shift(1).rolling(window=3, min_periods=1).mean()
df['GHI_roll6']      = df['GHI'].shift(1).rolling(window=6, min_periods=1).mean()
df['Temp_lag1']      = df['Temperature'].shift(1)

df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)
print(f"Dataset after feature engineering: {df.shape[0]} samples")

# =============================================================
# 2. Feature Sets
# =============================================================
BASE_FEATURES = [
    'Hour', 'Month', 'DayOfYear',
    'Temperature', 'Dew Point', 'Pressure',
    'Relative Humidity', 'Wind Speed', 'Wind Direction',
    'Hour_sin', 'Hour_cos', 'Month_sin', 'Month_cos',
]

PHASE2_FEATURES = BASE_FEATURES + [
    'GHI_lag1', 'GHI_lag2', 'GHI_lag3',
    'GHI_roll3', 'GHI_roll6',
    'Temp_lag1',
]

TARGET = 'GHI'

train_mask = df['Year'] <= 2013
test_mask  = df['Year'] == 2014

X_train = df.loc[train_mask, PHASE2_FEATURES].copy()
X_test  = df.loc[test_mask,  PHASE2_FEATURES].copy()
y_train = df.loc[train_mask, TARGET].copy()
y_test  = df.loc[test_mask,  TARGET].copy()

print(f"\nTrain set : {X_train.shape[0]:,} samples (2010-2013)")
print(f"Test set  : {X_test.shape[0]:,} samples (2014)")
print(f"Features  : {len(PHASE2_FEATURES)} ({len(BASE_FEATURES)} base + 6 lag/rolling)")

# =============================================================
# 3. Evaluation Helper
# =============================================================
def evaluate_model(y_true, y_pred, label=""):
    r2   = r2_score(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    nrmse = rmse / (y_true.max() - y_true.min()) * 100
    daytime_mask = y_true > 0
    mape = np.mean(np.abs(
        (y_true[daytime_mask] - y_pred[daytime_mask]) / y_true[daytime_mask]
    )) * 100
    if label:
        print(f"  R^2: {r2:.4f} | MAE: {mae:.2f} W/m2 | RMSE: {rmse:.2f} W/m2 | "
              f"nRMSE: {nrmse:.2f}% | MAPE: {mape:.2f}%")
    return {'R2': r2, 'MAE': mae, 'RMSE': rmse, 'nRMSE': nrmse, 'MAPE': mape}

# =============================================================
# 4. Model Definitions & Training
# =============================================================
tscv = TimeSeriesSplit(n_splits=5)

results    = {}
cv_results = {}
predictions = {}
train_times = {}

# ── 4a. Random Forest ─────────────────────────────────────────
print("\n" + "-" * 65)
print("  [1/4] Random Forest Regressor")
print("-" * 65)
t0 = time.time()
rf = RandomForestRegressor(
    n_estimators=200, max_depth=20,
    min_samples_split=10, min_samples_leaf=5,
    max_features='sqrt', random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)
train_times['Random Forest'] = time.time() - t0

y_pred_rf = rf.predict(X_test)
predictions['Random Forest'] = y_pred_rf
cv_rf = cross_val_score(rf, X_train, y_train, cv=tscv, scoring='r2', n_jobs=-1)
cv_results['Random Forest'] = cv_rf
results['Random Forest'] = evaluate_model(y_test, y_pred_rf, "RF Test")
print(f"  CV R2: {cv_rf.mean():.4f} +/- {cv_rf.std():.4f} | Train time: {train_times['Random Forest']:.1f}s")

# ── 4b. Gradient Boosting ─────────────────────────────────────
print("\n" + "-" * 65)
print("  [2/4] Gradient Boosting Regressor")
print("-" * 65)
t0 = time.time()
gb_param_dist = {
    'n_estimators':     [100, 200, 300],
    'max_depth':        [4, 5, 6, 7],
    'learning_rate':    [0.05, 0.1, 0.15, 0.2],
    'subsample':        [0.7, 0.8, 0.9, 1.0],
    'min_samples_leaf': [5, 10, 20],
}
gb_base = GradientBoostingRegressor(random_state=42)
gb_search = RandomizedSearchCV(
    gb_base, gb_param_dist,
    n_iter=15, cv=TimeSeriesSplit(n_splits=3),
    scoring='r2', random_state=42, n_jobs=-1, verbose=0
)
gb_search.fit(X_train, y_train)
gb = gb_search.best_estimator_
train_times['Gradient Boosting'] = time.time() - t0

y_pred_gb = gb.predict(X_test)
predictions['Gradient Boosting'] = y_pred_gb
cv_gb = cross_val_score(gb, X_train, y_train, cv=tscv, scoring='r2', n_jobs=-1)
cv_results['Gradient Boosting'] = cv_gb
results['Gradient Boosting'] = evaluate_model(y_test, y_pred_gb, "GB Test")
print(f"  Best params: {gb_search.best_params_}")
print(f"  CV R2: {cv_gb.mean():.4f} +/- {cv_gb.std():.4f} | Train time: {train_times['Gradient Boosting']:.1f}s")

# ── 4c. XGBoost ───────────────────────────────────────────────
if XGBOOST_AVAILABLE:
    print("\n" + "-" * 65)
    print("  [3/4] XGBoost Regressor")
    print("-" * 65)
    t0 = time.time()
    xgb_param_dist = {
        'n_estimators':     [100, 200, 300],
        'max_depth':        [4, 5, 6, 7],
        'learning_rate':    [0.05, 0.1, 0.15],
        'subsample':        [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'reg_alpha':        [0, 0.1, 0.5],
        'reg_lambda':       [1, 1.5, 2],
    }
    xgb_base = xgb.XGBRegressor(random_state=42, n_jobs=-1, verbosity=0)
    xgb_search = RandomizedSearchCV(
        xgb_base, xgb_param_dist,
        n_iter=15, cv=TimeSeriesSplit(n_splits=3),
        scoring='r2', random_state=42, n_jobs=-1, verbose=0
    )
    xgb_search.fit(X_train, y_train)
    xgb_model = xgb_search.best_estimator_
    train_times['XGBoost'] = time.time() - t0

    y_pred_xgb = xgb_model.predict(X_test)
    predictions['XGBoost'] = y_pred_xgb
    cv_xgb = cross_val_score(xgb_model, X_train, y_train, cv=tscv, scoring='r2', n_jobs=-1)
    cv_results['XGBoost'] = cv_xgb
    results['XGBoost'] = evaluate_model(y_test, y_pred_xgb, "XGB Test")
    print(f"  Best params: {xgb_search.best_params_}")
    print(f"  CV R2: {cv_xgb.mean():.4f} +/- {cv_xgb.std():.4f} | Train time: {train_times['XGBoost']:.1f}s")
else:
    print("\n  [3/4] XGBoost -- SKIPPED (not installed)")

# ── 4d. SVR ───────────────────────────────────────────────────
print("\n" + "-" * 65)
print("  [4/4] Support Vector Regressor (SVR)")
print("-" * 65)
print("  [Note] Training on 15k-sample subset for speed")
t0 = time.time()
SUBSET = 15000
idx_sub = np.random.RandomState(42).choice(len(X_train), size=min(SUBSET, len(X_train)), replace=False)
idx_sub.sort()
X_train_sub = X_train.iloc[idx_sub]
y_train_sub = y_train.iloc[idx_sub]

svr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svr',    SVR(kernel='rbf', C=500, gamma='scale', epsilon=10))
])
svr_pipeline.fit(X_train_sub, y_train_sub)
train_times['SVR'] = time.time() - t0

y_pred_svr = svr_pipeline.predict(X_test)
predictions['SVR'] = y_pred_svr
cv_svr = cross_val_score(svr_pipeline, X_train_sub, y_train_sub, cv=TimeSeriesSplit(n_splits=3), scoring='r2', n_jobs=-1)
cv_results['SVR'] = cv_svr
results['SVR'] = evaluate_model(y_test, y_pred_svr, "SVR Test")
print(f"  CV R2: {cv_svr.mean():.4f} +/- {cv_svr.std():.4f} | Train time: {train_times['SVR']:.1f}s")

# =============================================================
# 5. Summary Table
# =============================================================
print("\n" + "=" * 85)
print("  MODEL COMPARISON SUMMARY -- TEST SET (2014)")
print("=" * 85)
print(f"  {'Model':<22} {'R2':>8} {'MAE':>10} {'RMSE':>10} {'nRMSE':>9} {'MAPE':>9} {'CV R2':>14} {'Time(s)':>9}")
print("  " + "-" * 83)
for name, m in results.items():
    cv_mean = cv_results[name].mean()
    cv_std  = cv_results[name].std()
    t       = train_times[name]
    print(f"  {name:<22} {m['R2']:>8.4f} {m['MAE']:>9.2f}  {m['RMSE']:>9.2f}  "
          f"{m['nRMSE']:>8.2f}%  {m['MAPE']:>8.2f}%  "
          f"{cv_mean:.4f}+/-{cv_std:.4f}  {t:>7.1f}")
print("=" * 85)

best_model = max(results, key=lambda m: results[m]['R2'])

with open('phase2_results.txt', 'w') as f:
    f.write("PHASE 2: MODEL COMPARISON RESULTS\n")
    f.write("DOP: Forecasting of Renewable Resources Using AI\n")
    f.write("BITS Pilani Hyderabad | EEE Dept\n")
    f.write("=" * 65 + "\n\n")
    f.write(f"{'Model':<22} {'R2':>8} {'MAE':>10} {'RMSE':>10} {'nRMSE':>9} {'CV R2':>14}\n")
    f.write("-" * 65 + "\n")
    for name, m in results.items():
        cv_mean = cv_results[name].mean()
        cv_std  = cv_results[name].std()
        f.write(f"{name:<22} {m['R2']:>8.4f} {m['MAE']:>9.2f}  {m['RMSE']:>9.2f}  "
                f"{m['nRMSE']:>8.2f}%  {cv_mean:.4f}+/-{cv_std:.4f}\n")
    f.write("\nBest Model: " + best_model + "\n")
    f.write("\nFeatures used (Phase 2):\n")
    for feat in PHASE2_FEATURES:
        f.write(f"  - {feat}\n")
print("\nSaved: phase2_results.txt")

# =============================================================
# 6. Plots
# =============================================================
active_models = list(results.keys())
active_colors = [MODEL_COLORS[m] for m in active_models]

# PLOT 15: Metrics Comparison Bar Chart
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
metrics_to_plot = [('R2', True), ('MAE', False), ('nRMSE', False)]

for ax, (metric, higher_is_better) in zip(axes, metrics_to_plot):
    vals = [results[m][metric] for m in active_models]
    bars = ax.bar(active_models, vals, color=active_colors, edgecolor='white', width=0.55, zorder=3)
    ax.set_title(f'{metric} (Test Set 2014)', fontweight='bold')
    ax.set_ylabel(metric + (' Score' if metric == 'R2' else ' (W/m2)' if metric == 'MAE' else ' (%)'))
    ax.set_xticklabels(active_models, rotation=20, ha='right', fontsize=9)
    ax.grid(axis='y', alpha=0.4, zorder=0)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(vals) * 0.01,
                f'{val:.4f}' if metric == 'R2' else f'{val:.2f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    if higher_is_better:
        ax.set_ylim(min(vals) * 0.995, max(vals) * 1.01)
    else:
        ax.set_ylim(0, max(vals) * 1.15)

plt.suptitle('Phase 2: Model Comparison -- Test Set Performance (2014)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/15_model_comparison_metrics.png', bbox_inches='tight')
plt.close()
print("Saved: plots/15_model_comparison_metrics.png")

# PLOT 16: Cross-Validation Box Plot
fig, ax = plt.subplots(figsize=(10, 6))
cv_data = [cv_results[m] for m in active_models]
bp = ax.boxplot(cv_data, labels=active_models, patch_artist=True,
                medianprops=dict(color='white', linewidth=2.5))
for patch, color in zip(bp['boxes'], active_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)
for whisker in bp['whiskers']:
    whisker.set(linewidth=1.5, linestyle='--', color='gray')
for cap in bp['caps']:
    cap.set(linewidth=1.5, color='gray')

ax.set_ylabel('R2 Score')
ax.set_title('Time-Series Cross-Validation R2 (5-Fold)', fontweight='bold')
ax.yaxis.grid(True, alpha=0.4)
for i, (m, cv) in enumerate(zip(active_models, cv_data), start=1):
    ax.text(i, min(cv) - 0.005,
            f'mu={np.mean(cv):.4f}\nsigma={np.std(cv):.4f}',
            ha='center', va='top', fontsize=8, color='#333333')
plt.tight_layout()
plt.savefig('plots/16_cv_comparison.png', bbox_inches='tight')
plt.close()
print("Saved: plots/16_cv_comparison.png")

# PLOT 17: 7-Day Forecast Comparison
fig, ax = plt.subplots(figsize=(16, 6))
test_df = df[test_mask].copy().reset_index(drop=True)
week_mask = (test_df['Month'] == 3) & (test_df['Day'] >= 10) & (test_df['Day'] <= 16)
week_idx  = test_df[week_mask].index
week_actual = test_df.loc[week_idx, 'GHI'].values
hours = range(len(week_actual))

ax.plot(hours, week_actual, '-', color='black', linewidth=2.5, label='Actual GHI', zorder=5)
for name in active_models:
    week_pred = predictions[name][week_idx]
    ax.plot(hours, week_pred, '--', color=MODEL_COLORS[name],
            linewidth=1.8, label=name, alpha=0.85)

for i in range(0, len(week_actual), 24):
    ax.axvline(x=i, color='gray', linestyle=':', alpha=0.3)

day_labels = ['Mar 10', 'Mar 11', 'Mar 12', 'Mar 13', 'Mar 14', 'Mar 15', 'Mar 16']
ax.set_xticks([i * 24 + 12 for i in range(7)])
ax.set_xticklabels(day_labels)
ax.set_xlabel('Date (March 2014)')
ax.set_ylabel('GHI (W/m2)')
ax.set_title('7-Day Forecast Comparison: All Models (March 10-16, 2014)', fontweight='bold')
ax.legend(fontsize=10, loc='upper right')
plt.tight_layout()
plt.savefig('plots/17_7day_forecast_comparison.png', bbox_inches='tight')
plt.close()
print("Saved: plots/17_7day_forecast_comparison.png")

# PLOT 18: Actual vs Predicted Scatter (All Models)
n_models = len(active_models)
fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
if n_models == 1:
    axes = [axes]

for ax, name in zip(axes, active_models):
    y_pred = predictions[name]
    r2  = results[name]['R2']
    mae = results[name]['MAE']
    ax.scatter(y_test, y_pred, alpha=0.07, s=4, color=MODEL_COLORS[name])
    lim = max(y_test.max(), max(y_pred))
    ax.plot([0, lim], [0, lim], 'r--', linewidth=1.5, label='Perfect')
    ax.set_xlabel('Actual GHI (W/m2)')
    ax.set_ylabel('Predicted GHI (W/m2)')
    ax.set_title(f'{name}\nR2={r2:.4f} | MAE={mae:.1f}', fontweight='bold')
    ax.legend(fontsize=9)

plt.suptitle('Actual vs Predicted GHI -- All Models (Test Set 2014)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/18_scatter_all_models.png', bbox_inches='tight')
plt.close()
print("Saved: plots/18_scatter_all_models.png")

# PLOT 19: Feature Importance (Tree Models)
tree_models = {}
if 'Random Forest' in results:
    tree_models['Random Forest'] = rf.feature_importances_
if 'Gradient Boosting' in results:
    tree_models['Gradient Boosting'] = gb.feature_importances_
if 'XGBoost' in results and XGBOOST_AVAILABLE:
    tree_models['XGBoost'] = xgb_model.feature_importances_

if len(tree_models) >= 1:
    n_tm = len(tree_models)
    fig, axes = plt.subplots(1, n_tm, figsize=(7 * n_tm, 7))
    if n_tm == 1:
        axes = [axes]
    for ax, (name, importances) in zip(axes, tree_models.items()):
        feat_series = pd.Series(importances, index=PHASE2_FEATURES).sort_values(ascending=True)
        feat_series.plot(kind='barh', ax=ax, color=MODEL_COLORS[name], edgecolor='white')
        ax.set_title(f'{name} Feature Importance', fontweight='bold')
        ax.set_xlabel('Importance Score')
        for i, val in enumerate(feat_series.values):
            ax.text(val + 0.001, i, f'{val:.3f}', va='center', fontsize=8)
    plt.suptitle('Phase 2: Feature Importance Comparison', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots/19_feature_importance_comparison.png', bbox_inches='tight')
    plt.close()
    print("Saved: plots/19_feature_importance_comparison.png")

# PLOT 20: Residual Distributions
fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5), sharey=False)
if n_models == 1:
    axes = [axes]

for ax, name in zip(axes, active_models):
    residuals = y_test.values - predictions[name]
    ax.hist(residuals, bins=60, color=MODEL_COLORS[name], edgecolor='white', alpha=0.82)
    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.axvline(residuals.mean(), color='orange', linestyle='--', linewidth=1.5,
               label=f'Mean: {residuals.mean():.1f}')
    ax.set_xlabel('Residual (W/m2)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'{name}\nResiduals (sigma={residuals.std():.1f})', fontweight='bold')
    ax.legend(fontsize=9)

plt.suptitle('Residual Error Distribution -- All Models (Test Set 2014)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/20_residuals_comparison.png', bbox_inches='tight')
plt.close()
print("Saved: plots/20_residuals_comparison.png")

# PLOT 21: Summary Card
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')
lines = [
    "  PHASE 2 MODEL COMPARISON -- PERFORMANCE SUMMARY",
    "  " + "=" * 55,
    f"  Dataset: 2010-2013 Train | 2014 Test | {len(PHASE2_FEATURES)} features",
    f"  Phase 2 additions: lag-1/2/3 GHI, 3h/6h rolling avg",
    "",
    f"  {'Model':<22} {'R2':>8} {'MAE':>10} {'nRMSE':>10} {'CV R2':>14}",
    "  " + "-" * 55,
]
for name, m in results.items():
    marker = " <-- BEST" if name == best_model else ""
    cv_str = f"{cv_results[name].mean():.4f}+/-{cv_results[name].std():.4f}"
    lines.append(f"  {name:<22} {m['R2']:>8.4f} {m['MAE']:>9.2f}  {m['nRMSE']:>8.2f}%  {cv_str:>14}{marker}")
lines += [
    "",
    f"  Best Model: {best_model}  (R2={results[best_model]['R2']:.4f})",
    f"  Literature RF baseline: ~28% nRMSE  | Our RF: {results['Random Forest']['nRMSE']:.2f}%",
]
ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes, fontsize=10.5,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='#0d1117', edgecolor='#4A90D9', alpha=0.97),
        color='#00ff88')
plt.tight_layout()
plt.savefig('plots/21_phase2_summary_card.png', bbox_inches='tight')
plt.close()
print("Saved: plots/21_phase2_summary_card.png")

# PLOT 22: Monthly R2 Breakdown
fig, ax = plt.subplots(figsize=(13, 5))
months_labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
x = np.arange(12)
width = 0.35

test_df_eval = df[test_mask].copy().reset_index(drop=True)
models_to_compare = ['Random Forest', best_model] if best_model != 'Random Forest' else ['Random Forest']

for i, name in enumerate(models_to_compare):
    monthly_r2 = []
    for m in range(1, 13):
        m_idx = test_df_eval['Month'] == m
        if m_idx.sum() > 0:
            r2 = r2_score(test_df_eval.loc[m_idx, 'GHI'], predictions[name][m_idx])
        else:
            r2 = 0
        monthly_r2.append(r2)
    offset = (i - (len(models_to_compare) - 1) / 2) * width
    ax.bar(x + offset, monthly_r2, width, label=name,
           color=MODEL_COLORS[name], alpha=0.8, edgecolor='white')

ax.set_xlabel('Month')
ax.set_ylabel('R2 Score')
ax.set_title('Monthly Model Accuracy Comparison -- Test Year 2014', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(months_labels)
ax.set_ylim(0, 1.05)
ax.legend()
ax.yaxis.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig('plots/22_monthly_r2_comparison.png', bbox_inches='tight')
plt.close()
print("Saved: plots/22_monthly_r2_comparison.png")

# =============================================================
# 8. Final Summary
# =============================================================
print("\n" + "=" * 65)
print("  [DONE] Phase 2 complete!")
print("=" * 65)
print(f"  Best model : {best_model}  (R2 = {results[best_model]['R2']:.4f})")
print(f"  Plots saved: plots/15 to plots/22 (8 new figures)")
print(f"  Results saved: phase2_results.txt")
print("=" * 65)
