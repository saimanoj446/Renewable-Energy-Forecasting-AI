"""
=============================================================================
Phase 9 Deliverable for Prof. Arup Ratan:
3D Parametric Response Surface Plots:
  1. 3D Surface of R² vs Lookback Window x and Forecast Horizon td
  2. 3D Surface of MSE (Mean Squared Error) vs Lookback Window x and Horizon td
=============================================================================
"""

import os
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata, RectBivariateSpline

# Base directory setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, 'sapec_optimal_output', 'sapec_optimal_sweep_matrix.csv')

if not os.path.exists(CSV_PATH):
    CSV_PATH = os.path.join(BASE_DIR, '..', 'kuala_lumpur', 'sapec_optimal_output', 'sapec_optimal_sweep_matrix.csv')

OUT_DIR_SAPEC = os.path.join(BASE_DIR, 'sapec_optimal_output')
OUT_DIR_PLOTS = os.path.join(BASE_DIR, '..', 'plots')
OUT_DIR_KL_PLOTS = os.path.join(BASE_DIR, 'plots')

for d in [OUT_DIR_SAPEC, OUT_DIR_PLOTS, OUT_DIR_KL_PLOTS]:
    os.makedirs(d, exist_ok=True)

df = pd.read_csv(CSV_PATH)
df['MSE'] = df['RMSE'] ** 2

lookbacks = np.sort(df['Lookback_x'].unique())
horizons  = np.sort(df['Horizon_td'].unique())

pivot_r2  = df.pivot(index='Horizon_td', columns='Lookback_x', values='R2')
pivot_mse = df.pivot(index='Horizon_td', columns='Lookback_x', values='MSE')

X_grid, Y_grid = np.meshgrid(lookbacks, horizons)
Z_r2_grid = pivot_r2.values
Z_mse_grid = pivot_mse.values

# Fine grid for smooth surface rendering
x_fine = np.linspace(lookbacks.min(), lookbacks.max(), 100)
y_fine = np.linspace(horizons.min(), horizons.max(), 100)
X_fine, Y_fine = np.meshgrid(x_fine, y_fine)

# Bivariate spline interpolation
spline_r2  = RectBivariateSpline(horizons, lookbacks, Z_r2_grid, kx=3, ky=3)
spline_mse = RectBivariateSpline(horizons, lookbacks, Z_mse_grid, kx=3, ky=3)

Z_r2_fine  = spline_r2(y_fine, x_fine)
Z_mse_fine = spline_mse(y_fine, x_fine)

# Optimal point
opt_row = df.loc[df['R2'].idxmax()]
opt_x  = opt_row['Lookback_x']
opt_td = opt_row['Horizon_td']
opt_r2 = opt_row['R2']
opt_mse = opt_row['MSE']

def save_to_all(fig, filename):
    for d in [OUT_DIR_SAPEC, OUT_DIR_PLOTS, OUT_DIR_KL_PLOTS]:
        p = os.path.join(d, filename)
        fig.savefig(p, dpi=300, bbox_inches='tight')
    print(f"  [SAVED] {filename} (DPI=300)")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1: 3D Surface of R² vs (x, td)
# ─────────────────────────────────────────────────────────────────────────────
fig1 = plt.figure(figsize=(11, 8.5))
ax1 = fig1.add_subplot(111, projection='3d')

# Smooth surface with 'viridis'
surf1 = ax1.plot_surface(X_fine, Y_fine, Z_r2_fine, cmap=cm.viridis,
                         alpha=0.88, edgecolor='none', antialiased=True,
                         rstride=2, cstride=2)

# Scatter empirical points
ax1.scatter(df['Lookback_x'], df['Horizon_td'], df['R2'],
            color='#DC2626', s=45, edgecolor='black', linewidth=0.8,
            alpha=0.95, label='Empirical Grid Points (36 combinations)')

# Bottom contour projection
z_min1 = df['R2'].min() - 0.05
ax1.contour(X_fine, Y_fine, Z_r2_fine, zdir='z', offset=z_min1,
            cmap=cm.viridis, alpha=0.55, levels=12)

# Highlight Global Maximum
ax1.scatter([opt_x], [opt_td], [opt_r2], color='#FBBF24', s=160,
            edgecolor='#B45309', linewidth=2.0, zorder=10)
ax1.text(opt_x + 5, opt_td + 2, opt_r2 + 0.02,
         f"Global Peak R² = {opt_r2:.4f}\n(x* = {int(opt_x)}m, td* = {int(opt_td)}m)",
         color='#92400E', fontweight='bold', fontsize=11,
         bbox=dict(boxstyle='round,pad=0.35', facecolor='#FEF3C7', edgecolor='#F59E0B', alpha=0.9))

# Labels and styling
ax1.set_xlabel('\nLookback Window x (minutes)', fontsize=12, fontweight='bold', labelpad=10)
ax1.set_ylabel('\nForecast Horizon td (minutes)', fontsize=12, fontweight='bold', labelpad=10)
ax1.set_zlabel('\nCoefficient of Determination (R²)', fontsize=12, fontweight='bold', labelpad=10)
ax1.set_title('3D Response Surface: Forecast R² vs (Lookback x, Horizon td)\n'
              'Dataset: Kuala Lumpur 1-Min Pyranometer | Oct-Dec 2025 Test Holdout',
              fontsize=13, fontweight='bold', pad=15)

ax1.set_xlim(15, 120)
ax1.set_ylim(5, 60)
ax1.set_zlim(z_min1, 0.90)

ax1.view_init(elev=28, azim=-125)
cb1 = fig1.colorbar(surf1, ax=ax1, shrink=0.58, aspect=14, pad=0.1)
cb1.set_label('R² Score', fontweight='bold', fontsize=11)
ax1.legend(loc='upper left', frameon=True, fontsize=10)

plt.tight_layout()
save_to_all(fig1, '79_ph9_3d_surface_r2_x_td.png')
plt.close(fig1)

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2: 3D Surface of MSE vs (x, td)
# ─────────────────────────────────────────────────────────────────────────────
fig2 = plt.figure(figsize=(11, 8.5))
ax2 = fig2.add_subplot(111, projection='3d')

# Smooth surface with reversed 'plasma' (lower MSE is better)
surf2 = ax2.plot_surface(X_fine, Y_fine, Z_mse_fine, cmap=cm.plasma,
                         alpha=0.88, edgecolor='none', antialiased=True,
                         rstride=2, cstride=2)

# Scatter empirical points
ax2.scatter(df['Lookback_x'], df['Horizon_td'], df['MSE'],
            color='#0284C7', s=45, edgecolor='black', linewidth=0.8,
            alpha=0.95, label='Empirical Grid Points (36 combinations)')

# Bottom contour projection
z_min2 = df['MSE'].min() - 3000
ax2.contour(X_fine, Y_fine, Z_mse_fine, zdir='z', offset=z_min2,
            cmap=cm.plasma, alpha=0.55, levels=12)

# Highlight Global Minimum
ax2.scatter([opt_x], [opt_td], [opt_mse], color='#10B981', s=160,
            edgecolor='#065F46', linewidth=2.0, zorder=10)
ax2.text(opt_x + 5, opt_td + 2, opt_mse + 1500,
         f"Global Min MSE = {opt_mse:,.0f} W²/m⁴\n(RMSE = {np.sqrt(opt_mse):.1f} W/m² at x*={int(opt_x)}m, td*={int(opt_td)}m)",
         color='#065F46', fontweight='bold', fontsize=11,
         bbox=dict(boxstyle='round,pad=0.35', facecolor='#D1FAE5', edgecolor='#10B981', alpha=0.9))

# Labels and styling
ax2.set_xlabel('\nLookback Window x (minutes)', fontsize=12, fontweight='bold', labelpad=10)
ax2.set_ylabel('\nForecast Horizon td (minutes)', fontsize=12, fontweight='bold', labelpad=10)
ax2.set_zlabel('\nMean Squared Error (MSE) [W²/m⁴]', fontsize=12, fontweight='bold', labelpad=10)
ax2.set_title('3D Response Surface: Forecast MSE vs (Lookback x, Horizon td)\n'
              'Dataset: Kuala Lumpur 1-Min Pyranometer | Oct-Dec 2025 Test Holdout',
              fontsize=13, fontweight='bold', pad=15)

ax2.set_xlim(15, 120)
ax2.set_ylim(5, 60)
ax2.set_zlim(z_min2, df['MSE'].max() + 2000)

ax2.view_init(elev=28, azim=-125)
cb2 = fig2.colorbar(surf2, ax=ax2, shrink=0.58, aspect=14, pad=0.1)
cb2.set_label('MSE (W²/m⁴)', fontweight='bold', fontsize=11)
ax2.legend(loc='upper left', frameon=True, fontsize=10)

plt.tight_layout()
save_to_all(fig2, '80_ph9_3d_surface_mse_x_td.png')
plt.close(fig2)

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 3: Dual-Panel Side-by-Side 3D Surfaces (R² and MSE together)
# ─────────────────────────────────────────────────────────────────────────────
fig3 = plt.figure(figsize=(19, 8.5))

# Subplot 1: R²
ax3a = fig3.add_subplot(121, projection='3d')
surf3a = ax3a.plot_surface(X_fine, Y_fine, Z_r2_fine, cmap=cm.viridis,
                           alpha=0.88, edgecolor='none', antialiased=True,
                           rstride=2, cstride=2)
ax3a.scatter(df['Lookback_x'], df['Horizon_td'], df['R2'],
             color='#DC2626', s=35, edgecolor='black', linewidth=0.7, alpha=0.9)
ax3a.contour(X_fine, Y_fine, Z_r2_fine, zdir='z', offset=z_min1,
             cmap=cm.viridis, alpha=0.5, levels=10)
ax3a.scatter([opt_x], [opt_td], [opt_r2], color='#FBBF24', s=140, edgecolor='#B45309', linewidth=2.0)
ax3a.text(opt_x + 5, opt_td + 2, opt_r2 + 0.02,
          f"Peak R² = {opt_r2:.4f}\n(x*={int(opt_x)}m, td*={int(opt_td)}m)",
          color='#92400E', fontweight='bold', fontsize=10,
          bbox=dict(boxstyle='round,pad=0.3', facecolor='#FEF3C7', edgecolor='#F59E0B', alpha=0.9))
ax3a.set_xlabel('\nLookback x (min)', fontsize=11, fontweight='bold', labelpad=8)
ax3a.set_ylabel('\nHorizon td (min)', fontsize=11, fontweight='bold', labelpad=8)
ax3a.set_zlabel('\nR² Score (Higher is Better)', fontsize=11, fontweight='bold', labelpad=8)
ax3a.set_title('(a) Accuracy Manifold: R² Response Surface', fontsize=13, fontweight='bold')
ax3a.set_xlim(15, 120); ax3a.set_ylim(5, 60); ax3a.set_zlim(z_min1, 0.90)
ax3a.view_init(elev=26, azim=-125)
cb3a = fig3.colorbar(surf3a, ax=ax3a, shrink=0.55, aspect=15, pad=0.08)
cb3a.set_label('R²', fontweight='bold')

# Subplot 2: MSE
ax3b = fig3.add_subplot(122, projection='3d')
surf3b = ax3b.plot_surface(X_fine, Y_fine, Z_mse_fine, cmap=cm.plasma,
                           alpha=0.88, edgecolor='none', antialiased=True,
                           rstride=2, cstride=2)
ax3b.scatter(df['Lookback_x'], df['Horizon_td'], df['MSE'],
             color='#0284C7', s=35, edgecolor='black', linewidth=0.7, alpha=0.9)
ax3b.contour(X_fine, Y_fine, Z_mse_fine, zdir='z', offset=z_min2,
             cmap=cm.plasma, alpha=0.5, levels=10)
ax3b.scatter([opt_x], [opt_td], [opt_mse], color='#10B981', s=140, edgecolor='#065F46', linewidth=2.0)
ax3b.text(opt_x + 5, opt_td + 2, opt_mse + 1500,
          f"Min MSE = {opt_mse:,.0f} W²/m⁴\n(RMSE = {np.sqrt(opt_mse):.1f} W/m²)",
          color='#065F46', fontweight='bold', fontsize=10,
          bbox=dict(boxstyle='round,pad=0.3', facecolor='#D1FAE5', edgecolor='#10B981', alpha=0.9))
ax3b.set_xlabel('\nLookback x (min)', fontsize=11, fontweight='bold', labelpad=8)
ax3b.set_ylabel('\nHorizon td (min)', fontsize=11, fontweight='bold', labelpad=8)
ax3b.set_zlabel('\nMSE [W²/m⁴] (Lower is Better)', fontsize=11, fontweight='bold', labelpad=8)
ax3b.set_title('(b) Error Manifold: Mean Squared Error (MSE) Surface', fontsize=13, fontweight='bold')
ax3b.set_xlim(15, 120); ax3b.set_ylim(5, 60); ax3b.set_zlim(z_min2, df['MSE'].max() + 2000)
ax3b.view_init(elev=26, azim=-125)
cb3b = fig3.colorbar(surf3b, ax=ax3b, shrink=0.55, aspect=15, pad=0.08)
cb3b.set_label('MSE (W²/m⁴)', fontweight='bold')

plt.suptitle('Coupled 3D Parametric Optimization Manifolds across Lookback (x) and Horizon (td)\n'
             'SAPEC 2027 — Track 9: Machine Learning & AI in Power Electronics | Supervisor: Prof. Arup Ratan',
             fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
save_to_all(fig3, '81_ph9_dual_3d_surface_r2_mse.png')
plt.close(fig3)

print("\n" + "="*70)
print("  [SUCCESS] All three 3D surface figures generated successfully!")
print("  - 79_ph9_3d_surface_r2_x_td.png")
print("  - 80_ph9_3d_surface_mse_x_td.png")
print("  - 81_ph9_dual_3d_surface_r2_mse.png")
print("="*70)
