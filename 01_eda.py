"""
=============================================================
Person 1: Data Engineer — Exploratory Data Analysis (EDA)
DOP: Forecasting of Renewable Resources Using AI
BITS Hyderabad | EEE Dept | Under Arup Ratan Sir
=============================================================
Dataset: BA_Combined.xlsx (Rajasthan Solar Irradiance, 2010-2014)
- 43,800 rows (5 years × 8,760 hours/year)
- 14 columns, 0 missing values
- Target: GHI (Global Horizontal Irradiance) in W/m²
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import seaborn as sns
import os

# ── Style Setup ──────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.facecolor': 'white',
})
sns.set_style("whitegrid")

# Create output directory
os.makedirs('plots', exist_ok=True)

# ── 1. Load Data ─────────────────────────────────────────────
print("Loading data...")
df = pd.read_excel('BA_Combined.xlsx', sheet_name='Sheet1')
print(f"Dataset: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Date range: {df['Year'].min()}-{df['Year'].max()}")
print(f"Missing values: {df.isnull().sum().sum()}")

# ── 2. Feature Engineering ───────────────────────────────────
# Create datetime column
df['DateTime'] = pd.to_datetime(
    df[['Year', 'Month', 'Day', 'Hour']].assign(Minute=30),
    format='%Y%m%d%H%M'
)
# Day of Year (1-365) — captures seasonal patterns
df['DayOfYear'] = df['DateTime'].dt.dayofyear

# Season labels
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Summer'
    elif month in [6, 7, 8, 9]:
        return 'Monsoon'
    else:
        return 'Post-Monsoon'

df['Season'] = df['Month'].apply(get_season)

# Is daytime (GHI > 0)
df['IsDaytime'] = (df['GHI'] > 0).astype(int)

# Save cleaned data as CSV for the model
df.to_csv('solar_data_cleaned.csv', index=False)
print("Saved cleaned data to solar_data_cleaned.csv")

# ── 3. Print Summary Statistics ──────────────────────────────
print("\n" + "="*60)
print("DATASET SUMMARY")
print("="*60)
print(f"{'Column':<20} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
print("-"*60)
for col in ['GHI', 'DNI', 'DHI', 'Temperature', 'Pressure', 'Relative Humidity', 'Wind Speed']:
    print(f"{col:<20} {df[col].mean():>10.2f} {df[col].std():>10.2f} {df[col].min():>10.2f} {df[col].max():>10.2f}")

# ── PLOT 1: GHI Distribution ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram of GHI
axes[0].hist(df['GHI'], bins=50, color='#FF8C00', edgecolor='white', alpha=0.85)
axes[0].set_xlabel('GHI (W/m²)')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Distribution of Global Horizontal Irradiance (GHI)')
axes[0].axvline(df['GHI'].mean(), color='red', linestyle='--', label=f"Mean: {df['GHI'].mean():.0f} W/m²")
axes[0].legend()

# GHI > 0 only (daytime)
daytime = df[df['GHI'] > 0]['GHI']
axes[1].hist(daytime, bins=50, color='#FFD700', edgecolor='white', alpha=0.85)
axes[1].set_xlabel('GHI (W/m²)')
axes[1].set_ylabel('Frequency')
axes[1].set_title('GHI Distribution (Daytime Only, GHI > 0)')
axes[1].axvline(daytime.mean(), color='red', linestyle='--', label=f"Mean: {daytime.mean():.0f} W/m²")
axes[1].legend()

plt.tight_layout()
plt.savefig('plots/01_ghi_distribution.png', bbox_inches='tight')
plt.close()
print("Saved: plots/01_ghi_distribution.png")

# ── PLOT 2: Hourly Average GHI Profile ───────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
hourly_avg = df.groupby('Hour')['GHI'].mean()
ax.fill_between(hourly_avg.index, hourly_avg.values, alpha=0.3, color='#FF6B35')
ax.plot(hourly_avg.index, hourly_avg.values, 'o-', color='#FF6B35', linewidth=2, markersize=6)
ax.set_xlabel('Hour of Day')
ax.set_ylabel('Average GHI (W/m²)')
ax.set_title('Average Solar Irradiance by Hour of Day (2010-2014)')
ax.set_xticks(range(24))
ax.set_xlim(-0.5, 23.5)
ax.axhline(0, color='gray', linewidth=0.5)

# Annotate peak
peak_hour = hourly_avg.idxmax()
peak_val = hourly_avg.max()
ax.annotate(f'Peak: {peak_val:.0f} W/m² at {peak_hour}:30',
            xy=(peak_hour, peak_val), xytext=(peak_hour + 2, peak_val + 30),
            arrowprops=dict(arrowstyle='->', color='red'),
            fontsize=11, color='red', fontweight='bold')

plt.tight_layout()
plt.savefig('plots/02_hourly_ghi_profile.png', bbox_inches='tight')
plt.close()
print("Saved: plots/02_hourly_ghi_profile.png")

# ── PLOT 3: Monthly Average GHI ──────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
monthly_avg = df.groupby('Month')['GHI'].mean()
months_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
colors = ['#4A90D9', '#4A90D9', '#FF6B35', '#FF6B35', '#FF6B35',
          '#2ECC71', '#2ECC71', '#2ECC71', '#2ECC71', '#E67E22', '#E67E22', '#4A90D9']
bars = ax.bar(range(1, 13), monthly_avg.values, color=colors, edgecolor='white', width=0.7)
ax.set_xticks(range(1, 13))
ax.set_xticklabels(months_labels)
ax.set_xlabel('Month')
ax.set_ylabel('Average GHI (W/m²)')
ax.set_title('Monthly Average Solar Irradiance')

# Add season legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#4A90D9', label='Winter'),
    Patch(facecolor='#FF6B35', label='Summer'),
    Patch(facecolor='#2ECC71', label='Monsoon'),
    Patch(facecolor='#E67E22', label='Post-Monsoon')
]
ax.legend(handles=legend_elements, loc='upper right')

# Value labels on bars
for bar, val in zip(bars, monthly_avg.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
            f'{val:.0f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('plots/03_monthly_ghi.png', bbox_inches='tight')
plt.close()
print("Saved: plots/03_monthly_ghi.png")

# ── PLOT 4: Correlation Heatmap ──────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))
corr_cols = ['GHI', 'DNI', 'DHI', 'Temperature', 'Dew Point',
             'Pressure', 'Relative Humidity', 'Wind Direction', 'Wind Speed', 'Hour', 'Month']
corr_matrix = df[corr_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdYlBu_r',
            center=0, square=True, linewidths=0.5, ax=ax,
            vmin=-1, vmax=1, cbar_kws={'shrink': 0.8})
ax.set_title('Feature Correlation Matrix', fontsize=14, pad=15)
plt.tight_layout()
plt.savefig('plots/04_correlation_heatmap.png', bbox_inches='tight')
plt.close()
print("Saved: plots/04_correlation_heatmap.png")

# ── PLOT 5: Seasonal GHI Comparison ─────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
season_order = ['Winter', 'Summer', 'Monsoon', 'Post-Monsoon']
season_colors = {'Winter': '#4A90D9', 'Summer': '#FF6B35', 'Monsoon': '#2ECC71', 'Post-Monsoon': '#E67E22'}
for season in season_order:
    mask = df['Season'] == season
    hourly = df[mask].groupby('Hour')['GHI'].mean()
    ax.plot(hourly.index, hourly.values, 'o-', label=season,
            color=season_colors[season], linewidth=2, markersize=4)
ax.set_xlabel('Hour of Day')
ax.set_ylabel('Average GHI (W/m²)')
ax.set_title('Seasonal Solar Irradiance Profiles')
ax.set_xticks(range(24))
ax.legend()
plt.tight_layout()
plt.savefig('plots/05_seasonal_ghi_profiles.png', bbox_inches='tight')
plt.close()
print("Saved: plots/05_seasonal_ghi_profiles.png")

# ── PLOT 6: Yearly Trend ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
yearly_monthly = df.groupby(['Year', 'Month'])['GHI'].mean().unstack(level=0)
yearly_monthly.plot(ax=ax, linewidth=2, marker='o', markersize=4)
ax.set_xlabel('Month')
ax.set_ylabel('Average GHI (W/m²)')
ax.set_title('Yearly Solar Irradiance Trends (2010-2014)')
ax.set_xticks(range(1, 13))
ax.set_xticklabels(months_labels)
ax.legend(title='Year')
plt.tight_layout()
plt.savefig('plots/06_yearly_trend.png', bbox_inches='tight')
plt.close()
print("Saved: plots/06_yearly_trend.png")

# ── PLOT 7: GHI vs Temperature Scatter ───────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
daytime_df = df[df['GHI'] > 0].sample(n=5000, random_state=42)
scatter = ax.scatter(daytime_df['Temperature'], daytime_df['GHI'],
                     c=daytime_df['Hour'], cmap='plasma', alpha=0.5, s=15)
ax.set_xlabel('Temperature (°C)')
ax.set_ylabel('GHI (W/m²)')
ax.set_title('Solar Irradiance vs Temperature (Colored by Hour)')
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Hour of Day')
plt.tight_layout()
plt.savefig('plots/07_ghi_vs_temperature.png', bbox_inches='tight')
plt.close()
print("Saved: plots/07_ghi_vs_temperature.png")

# ── PLOT 8: Irradiance Components (DHI, DNI, GHI) ───────────
fig, ax = plt.subplots(figsize=(12, 5))
hourly_components = df.groupby('Hour')[['GHI', 'DNI', 'DHI']].mean()
ax.fill_between(hourly_components.index, hourly_components['GHI'], alpha=0.2, color='#FF6B35')
ax.plot(hourly_components.index, hourly_components['GHI'], '-', label='GHI', color='#FF6B35', linewidth=2.5)
ax.plot(hourly_components.index, hourly_components['DNI'], '--', label='DNI', color='#4A90D9', linewidth=2)
ax.plot(hourly_components.index, hourly_components['DHI'], '-.', label='DHI', color='#2ECC71', linewidth=2)
ax.set_xlabel('Hour of Day')
ax.set_ylabel('Irradiance (W/m²)')
ax.set_title('Solar Irradiance Components by Hour')
ax.set_xticks(range(24))
ax.legend(fontsize=12)
plt.tight_layout()
plt.savefig('plots/08_irradiance_components.png', bbox_inches='tight')
plt.close()
print("Saved: plots/08_irradiance_components.png")

print("\n[DONE] EDA complete! All plots saved to plots/ directory.")
print(f"   Cleaned dataset saved: solar_data_cleaned.csv ({df.shape[0]} rows)")
