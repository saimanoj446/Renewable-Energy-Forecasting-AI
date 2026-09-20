"""
=============================================================
Person 2: ML Engineer — Random Forest Model
DOP: Forecasting of Renewable Resources Using AI
=============================================================
Model: Random Forest Regressor (scikit-learn)
Target: GHI (Global Horizontal Irradiance) in W/m²
Features: Temperature, Humidity, Pressure, Wind, Hour, Month, etc.
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import os
import warnings
warnings.filterwarnings('ignore')

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

#1. Load Cleaned Data
print("Loading data...")
df = pd.read_excel('BA_Combined.xlsx', sheet_name='Sheet1')

# Feature engineering
df['DayOfYear'] = pd.to_datetime(
    df[['Year', 'Month', 'Day']].assign(Hour=df['Hour'])
).dt.dayofyear

# Cyclical encoding for Hour and Month (helps RF capture periodicity)
df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)

print(f"Dataset loaded: {df.shape[0]} samples")

#2. Define Features and Target
feature_cols = [
    'Hour', 'Month', 'DayOfYear',
    'Temperature', 'Dew Point', 'Pressure',
    'Relative Humidity', 'Wind Speed', 'Wind Direction',
    'Hour_sin', 'Hour_cos', 'Month_sin', 'Month_cos'
]
target_col = 'GHI'

X = df[feature_cols].copy()
y = df[target_col].copy()

print(f"Features: {len(feature_cols)} columns")
print(f"Target: {target_col}")
print(f"Target stats — Mean: {y.mean():.2f}, Std: {y.std():.2f}, Max: {y.max()}")

#3. Time-Based Train/Test Split
# Use 2010-2013 for training, 2014 for testing
train_mask = df['Year'] <= 2013
test_mask = df['Year'] == 2014

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

print(f"\nTrain set: {X_train.shape[0]} samples (2010-2013)")
print(f"Test set:  {X_test.shape[0]} samples (2014)")

# 4. Train Random Forest
print("\nTraining Random Forest model...")
rf = RandomForestRegressor(
    n_estimators=200,       # 200 trees
    max_depth=20,           # Prevent overfitting
    min_samples_split=10,   # Minimum samples to split a node
    min_samples_leaf=5,     # Minimum samples in leaf
    max_features='sqrt',    # Use sqrt(n_features) for each split
    random_state=42,
    n_jobs=-1               # Use all CPU cores
)

rf.fit(X_train, y_train)
print("Model trained!")

# 5. Predictions
y_train_pred = rf.predict(X_train)
y_test_pred = rf.predict(X_test)

# 6. Evaluation Metrics
def evaluate(y_true, y_pred, label):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true[y_true > 0] - y_pred[y_true > 0]) / y_true[y_true > 0])) * 100
    nrmse = rmse / (y_true.max() - y_true.min()) * 100
    
    print(f"\n{'='*50}")
    print(f"  {label} Results")
    print(f"{'='*50}")
    print(f"  R² Score:           {r2:.4f}")
    print(f"  MAE:                {mae:.2f} W/m²")
    print(f"  RMSE:               {rmse:.2f} W/m²")
    print(f"  MAPE (daytime):     {mape:.2f}%")
    print(f"  nRMSE:              {nrmse:.2f}%")
    print(f"{'='*50}")
    
    return {'R²': r2, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'nRMSE': nrmse}

train_metrics = evaluate(y_train, y_train_pred, "TRAINING SET")
test_metrics = evaluate(y_test, y_test_pred, "TEST SET (2014)")

# 7. Cross-Validation
print("\nRunning 5-fold cross-validation...")
cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)
print(f"CV R² scores: {cv_scores}")
print(f"CV R² mean:   {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

#PLOT 9: Actual vs Predicted (Scatter)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Training
axes[0].scatter(y_train, y_train_pred, alpha=0.1, s=5, color='#4A90D9')
axes[0].plot([0, y_train.max()], [0, y_train.max()], 'r--', linewidth=2, label='Perfect Prediction')
axes[0].set_xlabel('Actual GHI (W/m²)')
axes[0].set_ylabel('Predicted GHI (W/m²)')
axes[0].set_title(f'Training Set (R² = {train_metrics["R²"]:.4f})')
axes[0].legend()

# Testing
axes[1].scatter(y_test, y_test_pred, alpha=0.15, s=5, color='#FF6B35')
axes[1].plot([0, y_test.max()], [0, y_test.max()], 'r--', linewidth=2, label='Perfect Prediction')
axes[1].set_xlabel('Actual GHI (W/m²)')
axes[1].set_ylabel('Predicted GHI (W/m²)')
axes[1].set_title(f'Test Set — 2014 (R² = {test_metrics["R²"]:.4f})')
axes[1].legend()

plt.suptitle('Random Forest: Actual vs Predicted GHI', fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig('plots/09_actual_vs_predicted_scatter.png', bbox_inches='tight')
plt.close()
print("Saved: plots/09_actual_vs_predicted_scatter.png")

# PLOT 10: Feature Importance
fig, ax = plt.subplots(figsize=(10, 7))
feat_imp = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=True)
colors_fi = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(feat_imp)))
feat_imp.plot(kind='barh', ax=ax, color=colors_fi, edgecolor='white')
ax.set_xlabel('Feature Importance (Gini Impurity)')
ax.set_title('Random Forest — Feature Importance')

# Annotate values
for i, (val, name) in enumerate(zip(feat_imp.values, feat_imp.index)):
    ax.text(val + 0.002, i, f'{val:.3f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('plots/10_feature_importance.png', bbox_inches='tight')
plt.close()
print("Saved: plots/10_feature_importance.png")

# ── PLOT 11: Actual vs Predicted Time Series (7-day sample) ───
fig, ax = plt.subplots(figsize=(14, 5))
# Pick a week from the test set (e.g., March 2014 — clear sky season)
test_df = df[test_mask].copy()
test_df['Predicted_GHI'] = y_test_pred
week_mask = (test_df['Month'] == 3) & (test_df['Day'] >= 10) & (test_df['Day'] <= 16)
week_data = test_df[week_mask]

hours = range(len(week_data))
ax.plot(hours, week_data['GHI'].values, '-', label='Actual GHI', color='#FF6B35', linewidth=2, alpha=0.8)
ax.plot(hours, week_data['Predicted_GHI'].values, '--', label='Predicted GHI', color='#4A90D9', linewidth=2, alpha=0.8)
ax.fill_between(hours, week_data['GHI'].values, week_data['Predicted_GHI'].values,
                alpha=0.15, color='gray', label='Error')
ax.set_xlabel('Hours (March 10-16, 2014)')
ax.set_ylabel('GHI (W/m²)')
ax.set_title('Solar Irradiance Forecast: 7-Day Sample (March 2014)')
ax.legend(fontsize=11)

# Add day markers
for i in range(0, len(week_data), 24):
    ax.axvline(x=i, color='gray', linestyle=':', alpha=0.3)

plt.tight_layout()
plt.savefig('plots/11_timeseries_7day.png', bbox_inches='tight')
plt.close()
print("Saved: plots/11_timeseries_7day.png")

# PLOT 12: Residual Analysis
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

residuals = y_test.values - y_test_pred

# Residual distribution
axes[0].hist(residuals, bins=60, color='#9B59B6', edgecolor='white', alpha=0.8)
axes[0].axvline(0, color='red', linestyle='--', linewidth=2)
axes[0].set_xlabel('Residual (Actual - Predicted) W/m²')
axes[0].set_ylabel('Frequency')
axes[0].set_title(f'Residual Distribution (Mean: {residuals.mean():.2f})')

# Residuals vs Predicted
axes[1].scatter(y_test_pred, residuals, alpha=0.1, s=5, color='#9B59B6')
axes[1].axhline(0, color='red', linestyle='--', linewidth=2)
axes[1].set_xlabel('Predicted GHI (W/m²)')
axes[1].set_ylabel('Residual (W/m²)')
axes[1].set_title('Residuals vs Predicted Values')

plt.tight_layout()
plt.savefig('plots/12_residual_analysis.png', bbox_inches='tight')
plt.close()
print("Saved: plots/12_residual_analysis.png")

# PLOT 13: Metrics Summary Card
fig, ax = plt.subplots(figsize=(8, 5))
ax.axis('off')

metrics_text = f"""
       RANDOM FOREST MODEL — PERFORMANCE SUMMARY
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    Model:        RandomForestRegressor (200 trees)
    Train:        2010–2013  |  Test: 2014
    Target:       GHI (Global Horizontal Irradiance)

    TRAIN SET (2010–2013):
      R² Score:    {train_metrics['R²']:.4f}
      MAE:         {train_metrics['MAE']:.2f} W/m²
      RMSE:        {train_metrics['RMSE']:.2f} W/m²
      nRMSE:       {train_metrics['nRMSE']:.2f}%

    TEST SET (2014):
      R² Score:    {test_metrics['R²']:.4f}
      MAE:         {test_metrics['MAE']:.2f} W/m²
      RMSE:        {test_metrics['RMSE']:.2f} W/m²
      nRMSE:       {test_metrics['nRMSE']:.2f}%

    CV R² (5-fold):   {cv_scores.mean():.4f} ± {cv_scores.std():.4f}
"""
ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='#1a1a2e', edgecolor='#4A90D9', alpha=0.95),
        color='#00ff88')
plt.tight_layout()
plt.savefig('plots/13_metrics_summary.png', bbox_inches='tight')
plt.close()
print("Saved: plots/13_metrics_summary.png")

# PLOT 14: Hourly Prediction Accuracy
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

test_df_eval = df[test_mask].copy()
test_df_eval['Predicted'] = y_test_pred
test_df_eval['Error'] = np.abs(test_df_eval['GHI'] - test_df_eval['Predicted'])

# MAE by Hour
hourly_mae = test_df_eval.groupby('Hour')['Error'].mean()
axes[0].bar(hourly_mae.index, hourly_mae.values, color='#E74C3C', alpha=0.8, edgecolor='white')
axes[0].set_xlabel('Hour of Day')
axes[0].set_ylabel('MAE (W/m²)')
axes[0].set_title('Prediction Error by Hour')
axes[0].set_xticks(range(24))

# R² by Month
monthly_r2 = []
for m in range(1, 13):
    m_mask = test_df_eval['Month'] == m
    if m_mask.sum() > 0:
        r2 = r2_score(test_df_eval.loc[m_mask, 'GHI'], test_df_eval.loc[m_mask, 'Predicted'])
        monthly_r2.append(r2)
    else:
        monthly_r2.append(0)

months_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
bars = axes[1].bar(range(1, 13), monthly_r2, color='#2ECC71', alpha=0.8, edgecolor='white')
axes[1].set_xlabel('Month')
axes[1].set_ylabel('R² Score')
axes[1].set_title('Model Accuracy by Month (2014)')
axes[1].set_xticks(range(1, 13))
axes[1].set_xticklabels(months_labels)
axes[1].set_ylim(0, 1)

for bar, val in zip(bars, monthly_r2):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{val:.2f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('plots/14_hourly_monthly_accuracy.png', bbox_inches='tight')
plt.close()
print("Saved: plots/14_hourly_monthly_accuracy.png")

# ── Save model metrics to file ────────────────────────────────
with open('model_results.txt', 'w') as f:
    f.write("RANDOM FOREST MODEL RESULTS\n")
    f.write("="*50 + "\n\n")
    f.write(f"Training Set (2010-2013):\n")
    f.write(f"  R²:    {train_metrics['R²']:.4f}\n")
    f.write(f"  MAE:   {train_metrics['MAE']:.2f} W/m²\n")
    f.write(f"  RMSE:  {train_metrics['RMSE']:.2f} W/m²\n")
    f.write(f"  nRMSE: {train_metrics['nRMSE']:.2f}%\n\n")
    f.write(f"Test Set (2014):\n")
    f.write(f"  R²:    {test_metrics['R²']:.4f}\n")
    f.write(f"  MAE:   {test_metrics['MAE']:.2f} W/m²\n")
    f.write(f"  RMSE:  {test_metrics['RMSE']:.2f} W/m²\n")
    f.write(f"  nRMSE: {test_metrics['nRMSE']:.2f}%\n\n")
    f.write(f"5-Fold Cross-Validation R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n")
    f.write(f"CV scores: {cv_scores}\n\n")
    f.write(f"Feature Importance:\n")
    for name, imp in feat_imp.sort_values(ascending=False).items():
        f.write(f"  {name:<25s} {imp:.4f}\n")
print("\nSaved: model_results.txt")

print("\n[DONE] Model training and evaluation complete!")
print("   All plots saved to plots/ directory")

# ── 8. Predict for a Specific Date
def predict_specific_date(year, month, day, df_source=df, model=rf, save_plot=True):
    """
    Extracts 24-hour weather data for a specific date and forecasts hourly GHI.
    
    Parameters:
        year (int): e.g. 2014
        month (int): 1 to 12
        day (int): 1 to 31
        df_source (DataFrame): dataset with features
        model (RandomForestRegressor): trained model
        save_plot (bool): whether to save a 24-hour forecast plot
        
    Returns:
        DataFrame containing Hourly Actual vs Predicted GHI and Errors.
    """
    date_mask = (df_source['Year'] == year) & (df_source['Month'] == month) & (df_source['Day'] == day)
    day_data = df_source[date_mask].copy().sort_values('Hour')
    
    if len(day_data) == 0:
        print(f"[ERROR] No data found for date: {year:04d}-{month:02d}-{day:02d}")
        return None
        
    # Extract features and predict
    X_date = day_data[feature_cols]
    day_data['Predicted_GHI'] = model.predict(X_date)
    day_data['Error'] = day_data['GHI'] - day_data['Predicted_GHI']
    day_data['Abs_Error'] = np.abs(day_data['Error'])
    
    date_str = f"{year:04d}-{month:02d}-{day:02d}"
    print("\n" + "="*65)
    print(f"  HOURLY SOLAR FORECAST FOR: {date_str}")
    print("="*65)
    print(f"{'Hour':<8} {'Temp(°C)':<10} {'Humidity(%)':<14} {'Actual GHI':<14} {'Predicted GHI':<15} {'Error':<10}")
    print("-"*65)
    for _, row in day_data.iterrows():
        print(f"{int(row['Hour']):02d}:30   {row['Temperature']:<10.1f} {row['Relative Humidity']:<14.1f} {row['GHI']:<14.1f} {row['Predicted_GHI']:<15.1f} {row['Error']:<+10.1f}")
    
    # Daily statistics
    actual_energy = day_data['GHI'].sum() / 1000  # kWh/m²
    pred_energy = day_data['Predicted_GHI'].sum() / 1000
    daily_mae = day_data['Abs_Error'].mean()
    daily_r2 = r2_score(day_data['GHI'], day_data['Predicted_GHI'])
    
    print("-"*65)
    print(f"Daily Actual Solar Energy:    {actual_energy:.2f} kWh/m²")
    print(f"Daily Predicted Solar Energy: {pred_energy:.2f} kWh/m²")
    print(f"Daily Mean Absolute Error:    {daily_mae:.2f} W/m²")
    print(f"Daily R² Score:               {daily_r2:.4f}")
    print("="*65)
    
    if save_plot:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(day_data['Hour'], day_data['GHI'], 'o-', color='#FF6B35', linewidth=2, label='Actual GHI')
        ax.plot(day_data['Hour'], day_data['Predicted_GHI'], 's--', color='#4A90D9', linewidth=2, label='Predicted GHI')
        ax.fill_between(day_data['Hour'], day_data['GHI'], day_data['Predicted_GHI'], alpha=0.2, color='gray', label='Error')
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('GHI (W/m²)')
        ax.set_title(f'24-Hour Solar Forecast: {date_str} (R² = {daily_r2:.4f}, MAE = {daily_mae:.1f} W/m²)')
        ax.set_xticks(range(0, 24, 2))
        ax.legend()
        plt.tight_layout()
        plot_path = f'plots/forecast_{date_str}.png'
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        print(f"Saved forecast plot: {plot_path}")
        
    return day_data[['Hour', 'Temperature', 'Relative Humidity', 'GHI', 'Predicted_GHI', 'Error', 'Abs_Error']]

# Example: Predict 24-hour solar irradiance for May 15, 2014 (Peak Summer)
print("\n--- Example: Specific Date Forecast ---")
sample_forecast = predict_specific_date(year=2014, month=5, day=15)
