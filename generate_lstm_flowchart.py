"""
=============================================================
Generate High-Level LSTM Architecture & Flowchart Diagram
DOP: Forecasting of Renewable Energy Resources Using AI
BITS Pilani Hyderabad | Prof. Arup Ratan
=============================================================
Outputs: plots/39_lstm_architecture_flowchart.png
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ArrowStyle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

OUTPUT_PATH = os.path.join(PLOTS_DIR, '39_lstm_architecture_flowchart.png')

# ── Figure Setup ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 9), dpi=300)
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')

# Color Palette (Dark modern tech aesthetic)
BG_COLOR       = '#0B132B'
PANEL_BG       = '#1C2541'
CARD_BG        = '#1E293B'
CARD_BORDER    = '#334155'
BLUE_ACCENT    = '#38BDF8'
ORANGE_ACCENT  = '#FB923C'
GREEN_ACCENT   = '#4ADE80'
PURPLE_ACCENT  = '#A78BFA'
TEXT_LIGHT     = '#F8FAFC'
TEXT_MUTED     = '#94A3B8'
YELLOW_ACCENT  = '#FACC15'

fig.patch.set_facecolor(BG_COLOR)

# ── Title & Subtitle ────────────────────────────────────────
ax.text(8.0, 8.55, "HIGH-LEVEL WORKING OF 2-LAYER PYTORCH LSTM SEQUENCE MODEL",
        color=TEXT_LIGHT, fontsize=17, fontweight='bold', ha='center', va='center')
ax.text(8.0, 8.22, "From 480 Input Data Points (60 min × 8 features) to Multi-Horizon Solar & Electrical Power Forecasts",
        color=BLUE_ACCENT, fontsize=11, fontweight='semibold', ha='center', va='center')

# Helper: Draw rounded card
def draw_card(x, y, w, h, bg_color, border_color, lw=1.5, radius=0.25):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle=f"round,pad={radius},rounding_size={radius}",
                         facecolor=bg_color, edgecolor=border_color, linewidth=lw, zorder=2)
    ax.add_patch(box)
    return box

# Helper: Draw smooth arrow
def draw_arrow(x1, y1, x2, y2, color=BLUE_ACCENT, lw=2.0, style='->'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=15, shrinkA=2, shrinkB=2), zorder=5)

# ═════════════════════════════════════════════════════════════
# STAGE 1: INPUT TENSOR (480 DATA POINTS)
# ═════════════════════════════════════════════════════════════
draw_card(0.6, 1.2, 3.0, 6.4, CARD_BG, ORANGE_ACCENT, lw=2.0)
ax.text(2.1, 7.3, "STAGE 1: INPUT DATA", color=ORANGE_ACCENT, fontsize=12, fontweight='bold', ha='center')
ax.text(2.1, 6.95, "480 Normalized Data Points", color=TEXT_LIGHT, fontsize=10, fontweight='semibold', ha='center')

# Subcard: Matrix Visual
draw_card(0.9, 3.8, 2.4, 2.8, '#0F172A', '#475569', lw=1.0)
ax.text(2.1, 6.3, "Lookback: 60 Minutes\n(T = 1 to 60)", color=YELLOW_ACCENT, fontsize=9.5, fontweight='bold', ha='center')
ax.text(2.1, 5.5, "×\n8 Physical Features:", color=TEXT_LIGHT, fontsize=9, ha='center', va='center')
features_text = "• GHI (Solar Flux)\n• Temperature\n• Direct (DNI)\n• Diffuse (DHI)\n• Rel. Humidity\n• Wind Speed\n• Air Pressure\n• Solar Zenith"
ax.text(1.1, 4.5, features_text, color=TEXT_MUTED, fontsize=8, va='center', linespacing=1.3)

# Subcard: PyTorch Tensor Shape
draw_card(0.9, 1.6, 2.4, 1.8, '#0F172A', ORANGE_ACCENT, lw=1.0)
ax.text(2.1, 3.0, "PyTorch Tensor Shape", color=TEXT_LIGHT, fontsize=9, fontweight='bold', ha='center')
ax.text(2.1, 2.4, "(Batch, 60, 8)", color=ORANGE_ACCENT, fontsize=13, fontweight='bold', ha='center')
ax.text(2.1, 1.9, "Batch Size = 64\nContinuous Rolling Window", color=TEXT_MUTED, fontsize=8, ha='center')

# Arrow from Stage 1 to Stage 2
draw_arrow(3.7, 4.4, 4.3, 4.4, color=ORANGE_ACCENT, lw=2.5)

# ═════════════════════════════════════════════════════════════
# STAGE 2: INSIDE THE LSTM CELL (GATES & CONVEYOR BELT)
# ═════════════════════════════════════════════════════════════
draw_card(4.4, 1.2, 6.8, 6.4, CARD_BG, BLUE_ACCENT, lw=2.2)
ax.text(7.8, 7.3, "STAGE 2: INSIDE THE LSTM RECURRENT CELL", color=BLUE_ACCENT, fontsize=12, fontweight='bold', ha='center')
ax.text(7.8, 6.95, "Uninterrupted Memory Conveyor Belt (Solves Vanishing Gradient)", color=TEXT_LIGHT, fontsize=10, ha='center')

# The Cell State Highway (Top Conveyor Belt)
draw_card(4.7, 5.5, 6.2, 1.2, '#0B132B', GREEN_ACCENT, lw=2.0)
ax.text(5.0, 6.1, "Ct-1", color=GREEN_ACCENT, fontsize=11, fontweight='bold', va='center')
ax.text(7.8, 6.35, "CELL STATE HIGHWAY (Long-Term Memory Conveyor Belt)", color=GREEN_ACCENT, fontsize=10.5, fontweight='bold', ha='center', va='center')
ax.text(7.8, 5.8, "Ct = ft ⊙ Ct-1  +  it ⊙ C~t    [Linear Addition Prevents Gradient Decay]", color=TEXT_LIGHT, fontsize=9, fontweight='semibold', ha='center', va='center')
ax.text(10.6, 6.1, "Ct", color=GREEN_ACCENT, fontsize=11, fontweight='bold', va='center')

# 3 Gate Boxes
# Gate 1: Forget Gate
draw_card(4.7, 3.4, 1.8, 1.7, '#0F172A', '#E11D48', lw=1.5)
ax.text(5.6, 4.85, "FORGET GATE", color='#FDA4AF', fontsize=9, fontweight='bold', ha='center')
ax.text(5.6, 4.4, "ft = σ(Wf·[h,x])", color=TEXT_LIGHT, fontsize=8.5, fontweight='bold', ha='center')
ax.text(5.6, 3.8, "Discards irrelevant\npast (e.g. night)", color=TEXT_MUTED, fontsize=7.5, ha='center')
draw_arrow(5.6, 5.1, 5.6, 5.5, color='#FDA4AF', lw=1.8)

# Gate 2: Input Gate + Candidate
draw_card(6.9, 3.4, 2.0, 1.7, '#0F172A', YELLOW_ACCENT, lw=1.5)
ax.text(7.9, 4.85, "INPUT GATE", color=YELLOW_ACCENT, fontsize=9, fontweight='bold', ha='center')
ax.text(7.9, 4.4, "it = σ(...) ⊙ tanh(...)", color=TEXT_LIGHT, fontsize=8.5, fontweight='bold', ha='center')
ax.text(7.9, 3.8, "Injects new cloud\nramp transients", color=TEXT_MUTED, fontsize=7.5, ha='center')
draw_arrow(7.9, 5.1, 7.9, 5.5, color=YELLOW_ACCENT, lw=1.8)

# Gate 3: Output Gate
draw_card(9.3, 3.4, 1.6, 1.7, '#0F172A', PURPLE_ACCENT, lw=1.5)
ax.text(10.1, 4.85, "OUTPUT GATE", color=PURPLE_ACCENT, fontsize=9, fontweight='bold', ha='center')
ax.text(10.1, 4.4, "ot = σ(Wo·[h,x])", color=TEXT_LIGHT, fontsize=8.5, fontweight='bold', ha='center')
ax.text(10.1, 3.8, "Filters short-term\nhidden state ht", color=TEXT_MUTED, fontsize=7.5, ha='center')
draw_arrow(10.1, 3.4, 10.1, 3.0, color=PURPLE_ACCENT, lw=1.8)

# Bottom: 2-Layer Stacked Architecture Note
draw_card(4.7, 1.5, 6.2, 1.5, '#0B132B', '#475569', lw=1.0)
ax.text(7.8, 2.65, "Stacked 2-Layer PyTorch Architecture in Code (08_lstm_forecast.py):", color=TEXT_LIGHT, fontsize=9, fontweight='bold', ha='center')
ax.text(7.8, 2.2, "Layer 1: LSTM (8 in → 64 hidden)   |   Dropout (p = 0.20)   |   Layer 2: LSTM (64 in → 64 hidden)", color=BLUE_ACCENT, fontsize=8.5, ha='center')
ax.text(7.8, 1.75, "Step t=60 Final Hidden Vector:  h60 ∈ ℝ^64  (Full sequence summary)", color=YELLOW_ACCENT, fontsize=8.5, fontweight='bold', ha='center')

# Arrow from Stage 2 to Stage 3
draw_arrow(11.3, 4.4, 11.9, 4.4, color=BLUE_ACCENT, lw=2.5)

# ═════════════════════════════════════════════════════════════
# STAGE 3: MULTI-HORIZON OUTPUTS & PV POWER
# ═════════════════════════════════════════════════════════════
draw_card(12.0, 1.2, 3.4, 6.4, CARD_BG, GREEN_ACCENT, lw=2.0)
ax.text(13.7, 7.3, "STAGE 3: FORECAST HEAD", color=GREEN_ACCENT, fontsize=12, fontweight='bold', ha='center')
ax.text(13.7, 6.95, "Multi-Horizon & Electrical Power", color=TEXT_LIGHT, fontsize=10, fontweight='semibold', ha='center')

# Multi-Horizon Box
draw_card(12.2, 3.8, 3.0, 2.8, '#0F172A', BLUE_ACCENT, lw=1.2)
ax.text(13.7, 6.3, "Linear Projection Head", color=TEXT_LIGHT, fontsize=9.5, fontweight='bold', ha='center')
ax.text(13.7, 5.9, "ŷ = W·h60 + b  (5 Horizons)", color=BLUE_ACCENT, fontsize=8.5, fontweight='bold', ha='center')

horizons_info = (
    "+1 min:   R²=0.898  (Skill: +3.1%)\n"
    "+5 min:   R²=0.749  (Skill: +11.7%)\n"
    "+15 min: R²=0.632  (Skill: +17.2%)\n"
    "+30 min: R²=0.571  (Skill: +20.3%)\n"
    "+60 min: R²=0.527  (Skill: +25.8% - Best)"
)
ax.text(12.4, 4.6, horizons_info, color=TEXT_LIGHT, fontsize=8, va='center', linespacing=1.35)

# Physics-Based MPP Box
draw_card(12.2, 1.5, 3.0, 2.0, '#0F172A', GREEN_ACCENT, lw=1.2)
ax.text(13.7, 3.2, "Physics-Based PV Power", color=GREEN_ACCENT, fontsize=9.5, fontweight='bold', ha='center')
ax.text(13.7, 2.7, "Sandia/IEC Cell Derating:", color=TEXT_MUTED, fontsize=8, ha='center')
ax.text(13.7, 2.25, "Pmpp = Pstc·(GHI/1000)·[1 + α(Tc - 25)]", color=YELLOW_ACCENT, fontsize=7.5, fontweight='bold', ha='center')
ax.text(13.7, 1.8, "Daytime Power R² = 0.9020", color=GREEN_ACCENT, fontsize=9.5, fontweight='bold', ha='center')

# ── Footer / Watermark ─────────────────────────────────────
ax.text(8.0, 0.45, "BITS Pilani Hyderabad Campus • Department of Electrical & Electronics Engineering • Supervised by Prof. Arup Ratan",
        color='#64748B', fontsize=9, ha='center', va='center')

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
plt.close()

print(f"SUCCESS: Flowchart generated at: {OUTPUT_PATH}")
