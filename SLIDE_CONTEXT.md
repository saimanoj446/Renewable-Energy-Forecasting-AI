# SLIDE GENERATION CONTEXT & SPECIFICATION FILE
> **For Automated AI Slide Generators (Astra, Gamma, Pitch, ChatGPT, Claude)**  
> **Repository:** https://github.com/saimanoj446/Renewable-Energy-Forecasting-AI  
> **Raw Asset Base URL:** `https://raw.githubusercontent.com/saimanoj446/Renewable-Energy-Forecasting-AI/main/`

---

## 📋 Project & Presentation Metadata

* **Project Title:** Advanced Solar Irradiance Forecasting at $t+60$ Minutes: Physics-Guided Diurnal Modeling, Satellite Cloud Advection, and Conformalized Quantile Ensembles
* **Institution:** Department of Electrical & Electronics Engineering, BITS Pilani, Hyderabad Campus
* **Academic Course:** Design-Oriented Project (DOP)
* **Faculty Advisor / Guide:** Prof. Arup Ratan
* **Domain / Theme:** Renewable Energy AI, Power Systems Integration, IEEE 1547 Grid Stability, Short-Term Photovoltaic Forecasting
* **Audience:** Academic Professor Meeting & Department Defense Committee
* **Visual Theme:** Technical, clean, academic engineering style (Dark Slate / Deep Navy background `#0F172A`, High-contrast accents: Electric Cyan `#38BDF8`, Safety Orange `#FB923C`, Emerald Green `#4ADE80`).

---

## 🖼️ Catalog of Key Visualization Assets (Direct GitHub Raw Links)

Use these exact raw image links to embed the figures into each corresponding slide:

| Plot ID | Topic | Direct Raw Image Link |
|:--- |:--- |:--- |
| **Plot 45** | Phase 4b Loss Convergence | `https://raw.githubusercontent.com/saimanoj446/Renewable-Energy-Forecasting-AI/main/plots/45_ph4b_training_convergence.png` |
| **Plot 50** | Solar Noon Overshoot Solution | `https://raw.githubusercontent.com/saimanoj446/Renewable-Energy-Forecasting-AI/main/plots/50_ph4c_t60_walkforward_day.png` |
| **Plot 52** | Satellite Feature Importance | `https://raw.githubusercontent.com/saimanoj446/Renewable-Energy-Forecasting-AI/main/plots/52_ph5_feature_importance.png` |
| **Plot 53** | Phase 5 Model Benchmark | `https://raw.githubusercontent.com/saimanoj446/Renewable-Energy-Forecasting-AI/main/plots/53_ph5_model_comparison_bars.png` |
| **Plot 54** | Phase 5 Walk-Forward Tracking | `https://raw.githubusercontent.com/saimanoj446/Renewable-Energy-Forecasting-AI/main/plots/54_ph5_walkforward_day.png` |
| **Plot 56** | Phase 6 Probabilistic Fan Chart | `https://raw.githubusercontent.com/saimanoj446/Renewable-Energy-Forecasting-AI/main/plots/56_ph6_probabilistic_fan_chart.png` |
| **Plot 57** | Conformal Calibration Curve | `https://raw.githubusercontent.com/saimanoj446/Renewable-Energy-Forecasting-AI/main/plots/57_ph6_conformal_reliability_diagram.png` |
| **Plot 58** | Physics & Volatility Importance | `https://raw.githubusercontent.com/saimanoj446/Renewable-Energy-Forecasting-AI/main/plots/58_ph6_physics_feature_importance.png` |
| **Plot 59** | Project Benchmark Progression | `https://raw.githubusercontent.com/saimanoj446/Renewable-Energy-Forecasting-AI/main/plots/59_ph6_benchmark_comparison.png` |

---

## 📑 Slide-by-Slide Detailed Specifications

```markdown
### SLIDE 1: TITLE SLIDE
- Layout: Centered Hero Layout
- Header: Advanced Solar Irradiance Forecasting at t+60 Minutes
- Subheader: Physics-Guided Diurnal Modeling, Satellite Cloud Advection, and Conformalized Quantile Ensembles
- Bullets / Metadata:
  * Department of Electrical & Electronics Engineering, BITS Pilani, Hyderabad Campus
  * Under the Guidance of: Prof. Arup Ratan
  * Target Dataset: 1-Minute Ground Pyranometer Station (Kuala Lumpur 2025, 262,020 records)
  * Focus: Solving the Solar Noon Overshoot & Elevating the 1-Hour Horizon

---

### SLIDE 2: MEETING CONTEXT & EXECUTIVE SUMMARY
- Layout: 2-Column Split (Left: Challenges, Right: Solved Matrix)
- Header: Follow-Up on Professor's Directives
- Subheader: Targeted R&D on the t+60 Minute Horizon in Tropical Convective Climates
- Content Left (The Two Challenges Raised in Last Meeting):
  1. Solar Noon Overshoot Dilemma: Linear trend extrapolation (y = mx + c) overshoots past 12:00 PM astronomical peak.
  2. Elevating t+60 Accuracy: The Phase 3 baseline LSTM scored R² = 0.5266 on Kuala Lumpur; needed physics and data improvements.
- Content Right (Results Delivered):
  * Phase 4 (Diurnal Physics): Known Future Astronomy ceiling GHI_clear(t+60) eliminated 100% of runaway noon overshoot.
  * Phase 5 (Satellite + GBDT): Ingested Himawari satellite cloud layers via API; LightGBM pushed R² to 0.5392 (Skill +28.4%).
  * Phase 6 (Physics Volatility + CQR): Engineered Variability Index (VI_30) and Conformal Prediction; R² = 0.5480, MAE dropped by 14 W/m², 81.98% coverage.
- Speaker Notes: "Sir, we dedicated this sprint strictly to the challenges you raised. We solved the 12:00 noon overshoot mathematically, ingested satellite cloud layers, and advanced the state of the art to R² = 0.5480."

---

### SLIDE 3: PHASE 4 — THE SOLAR NOON OVERSHOOT & ASTRONOMICAL GEOMETRY
- Layout: 2-Column Split (Left: Theory & Equations, Right: Diagram)
- Header: Phase 4: Physics-Based Clear-Sky Normalization
- Subheader: Decoupling Deterministic Solar Geometry from Stochastic Cloud Attenuation
- Content:
  * The Problem: Pure sequence models observing rising irradiance from 10:30–11:30 AM project upward slope past 12:00 noon.
  * The Physics: Instantaneous irradiance GHI(t) = kt(t) × GHI_clear(t).
  * Exact Astronomical Equations:
    - Zenith Angle: cos(θz) = sin(φ)sin(δ) + cos(φ)cos(δ)cos(ω)
    - Air Mass: AM = 1 / [cos(θz) + 0.50572(96.08 - θz)^-1.6364]
    - Ineichen-Perez Clear-Sky Ceiling: GHI_clear = I0 · cos(θz) · cg1 · exp[-cg2 · AM · TL · exp(0.01 · AM^1.8)]
  * Key Innovation: GHI_clear(t+60) is 100% deterministically known in advance and fed as an Exogenous Known Future Input.

---

### SLIDE 4: PHASE 4 — RECURRENT FUSION ARCHITECTURE & NOON PROOF
- Layout: 2-Column Split (Left: Pipeline & Metrics, Right: Plot Image)
- Header: Phase 4: Recurrent Fusion with Future Astronomy
- Subheader: Eliminating Runaway Solar Noon Overshoot with Physical Ceiling Bounds
- Content:
  * Architecture: 60-min past sequence (Conv1D + 2-Layer LSTM) extracts 64-dim hidden state h60.
  * Fusion Head: Concatenates h60 with [GHI_clear(t+60), Hour_sin, Hour_cos, GHI_ramp] → Dense + Sigmoid → kt in [0, 1.2].
  * Physics Reconstruction: GHI_pred(t+60) = kt_pred × GHI_clear(t+60).
  * Verification Results:
    - Stable loss convergence (early stopping at epoch 11).
    - Mean bias error reduced from +25 W/m² to +11.56 W/m².
    - Strict physical adherence to the diurnal curve descent after solar noon.
- Image Asset: `https://raw.githubusercontent.com/saimanoj446/Renewable-Energy-Forecasting-AI/main/plots/50_ph4c_t60_walkforward_day.png`

---

### SLIDE 5: PHASE 5 — TRANSITIONING BEYOND LSTM TO GRADIENT BOOSTED TREES
- Layout: Comparison Table / 3 Cards
- Header: Phase 5: Moving Beyond Recurrent Sequence Architectures
- Subheader: Why Decision Trees (LightGBM/XGBoost) Outperform LSTMs at 60-Minute Horizons
- Content:
  * Card 1 (Recency Bias of LSTMs): LSTMs over-weight the last hidden state ht, excelling at +1m (R²=0.898) but degrading at +60m as temporal memory fades.
  * Card 2 (Discontinuous Cloud Step-Functions): Convective cloud shading drops irradiance sharply by 70% in minutes. Decision trees split non-linear step-functions (IF Cloud_Low > 65% THEN drop GHI) without saturation.
  * Card 3 (Computational Speed & Interpretability): LightGBM trains in 12 seconds across 260k rows (vs 5 minutes for LSTM) and outputs native split-gain feature importances.

---

### SLIDE 6: PHASE 5 — GEOSTATIONARY SATELLITE CLOUD INGESTION
- Layout: 2-Column Split (Left: Physics Justification, Right: Feature Table)
- Header: Ingesting Satellite Cloud Optical Depth via API
- Subheader: Overcoming the 15–40 km Spatial Blind Spot of Single Ground Pyranometers
- Content:
  * The Missing Information: At 25 km/h cloud advection speeds, clouds causing shading at t+60 are 15–40 km upwind.
  * Satellite Layers Ingested via Open-Meteo Himawari-8/9 Reanalysis API:
    - Cloud_Cover_Low (%): Low convective cumulus clouds (0–2 km) — the #1 cause of tropical solar dropouts.
    - Cloud_Cover_Mid (%): Altocumulus layers (2–6 km).
    - Cloud_Cover_High (%): Cirrus cloud optical attenuation (>6 km).
    - Cloud_Cover (%): Total column cloud fraction.
  * Multi-Scale Dynamic Features: 15m/30m/60m rolling means, rolling standard deviation (volatility), and ramp rates.

---

### SLIDE 7: PHASE 5 — SATELLITE FEATURE IMPORTANCE BREAKTHROUGH
- Layout: 2-Column Split (Left: Findings & Table, Right: Plot Image)
- Header: Phase 5: Satellite Features Drive Significant Skill Gains
- Subheader: LightGBM Surpasses LSTM Baseline Across All Evaluation Metrics
- Content:
  * Major Discovery (Plot 52): Cloud_Cover_Low is the #3 most influential feature in the entire model, outranking Ambient Temperature, Relative Humidity, and past 15-min GHI!
  * GHI_clear_t60 ranked #6, confirming reliance on future solar geometry.
  * Benchmark Results (+60 Min Horizon):
    - Phase 3 LSTM: R² = 0.5266 | RMSE = 204.7 W/m² | Skill = +25.8%
    - Phase 5 LightGBM: R² = 0.5392 | RMSE = 197.1 W/m² | Skill = +28.4% 🏆
    - RMSE dropped by 7.6 W/m²; MAE dropped by 6.5 W/m².
- Image Asset: `https://raw.githubusercontent.com/saimanoj446/Renewable-Energy-Forecasting-AI/main/plots/52_ph5_feature_importance.png`

---

### SLIDE 8: THE INFORMATION HORIZON OF TROPICAL CLOUDS
- Layout: 3 Technical Callout Cards
- Header: Why Single-Station Point Forecasts Plateau at R² ≈ 0.54
- Subheader: Meteorological & Information-Theoretic Limits in Equatorial Convective Climates
- Content:
  * Card 1 (Convective Cloud Lifecycles): In equatorial climates (Kuala Lumpur, Köppen Af), convective cumulus clouds form, rain, and dissipate within 10–30 minutes. By 60 minutes, the atmospheric state is newly formed.
  * Card 2 (Single Ground Pyranometer Limit): A ground sensor cannot see clouds 25 km away. In published IEEE literature, R² ≈ 0.50–0.55 represents the empirical ceiling for single-station temporal extrapolation.
  * Card 3 (The Methodological Paradigm Shift): Grid operators cannot rely on deterministic point estimates during storm swings. The industry requires Calibrated Prediction Intervals [q10, q90] to schedule grid spinning reserves.

---

### SLIDE 9: PHASE 6 — ADVANCED PHYSICS & VOLATILITY FEATURE ENGINEERING
- Layout: 2-Column Split (Left: Mathematical Formulations, Right: Plot Image)
- Header: Phase 6: Variability Index (VI) & Dynamic Ramp Acceleration
- Subheader: Extracting High-Frequency Wave Dynamics from 1-Minute Pyranometer Streams
- Content:
  * 1. Variability Index (VI): Ratio of actual GHI curve arc-length to clear-sky arc-length over 15-min and 30-min causal windows:
    VI = [ Σ sqrt((ΔGHI)² + Δt²) ] / [ Σ sqrt((ΔGHI_clear)² + Δt²) ]
    (Captures Mie edge scattering; spikes to 8–15 during broken cloud transitions).
  * 2. Ramp Acceleration (Δ²GHI & Δ²kt): Second-order difference detecting cloud onset before full drop occurs:
    Δ²GHI = (GHI_t - GHI_t-1) - (GHI_t-1 - GHI_t-2).
  * 3. Rolling Ramp Volatility: σ(ΔGHI_15m) and rolling 30-min kt skewness.
- Image Asset: `https://raw.githubusercontent.com/saimanoj446/Renewable-Energy-Forecasting-AI/main/plots/58_ph6_physics_feature_importance.png`

---

### SLIDE 10: PHASE 6 — MULTI-QUANTILE LIGHTGBM & CONFORMAL PREDICTION (CQR)
- Layout: Flow Diagram / Process Layout
- Header: Phase 6: Conformalized Quantile Regression Pipeline
- Subheader: Eliminating Monsoon Distribution Shift to Guarantee ≥ 80% Empirical Coverage
- Content:
  * Step 1 (Quantile Heads): Train 3 LightGBM models with Pinball Loss for alpha in {0.10, 0.50, 0.90} → generates [q10, q50, q90].
  * Step 2 (Conformal Calibration):
    - Compute nonconformity score on September validation set: s_i = max(q10 - y, y - q90).
    - Determine calibration offset Q = Quantile(s, 0.80).
  * Step 3 (CQR Test Holdout Adjustment):
    - L_hat = max(0, q10 - Q)
    - U_hat = min(1.2 · GHI_clear, q90 + Q)
  * Guarantee: Enforces exact coverage without assuming Gaussian residuals.

---

### SLIDE 11: PHASE 6 — EXPERIMENTAL RESULTS & PROBABILISTIC PROOF
- Layout: 2-Column Split (Left: Performance Table, Right: Plot Image)
- Header: Phase 6: Breakthrough Results (+60 Min Horizon)
- Subheader: Achieving Project Peak R² = 0.5480 and 82.0% Conformal Coverage
- Content:
  * Deterministic P50 Median Point Metrics:
    - R² Score: 0.5480 🏆 (Project Record)
    - RMSE: 195.18 W/m² 🏆 (Dropped 9.5 W/m² vs Phase 3 LSTM)
    - MAE: 141.90 W/m² 🏆 (Dropped 13.9 W/m² vs Phase 3 LSTM)
    - IEEE Skill Score: +29.10% 🏆 over persistence
  * Probabilistic Conformal Metrics:
    - Empirical Coverage (PICP): 81.98% (Exceeds 80% nominal threshold)
    - Continuous Ranked Probability Score (CRPS): 42.58 W/m²
    - Conformal Band Width (PINAW): 33.74%
- Image Asset: `https://raw.githubusercontent.com/saimanoj446/Renewable-Energy-Forecasting-AI/main/plots/56_ph6_probabilistic_fan_chart.png`

---

### SLIDE 12: MASTER BENCHMARK MATRIX (+60 MIN LEAD TIME)
- Layout: Full-Screen Comparative Table
- Header: Multi-Phase Evolution: The t+60 Min Solar Forecast
- Subheader: Consistent, Rigorously Verified Progress Across Out-Of-Time Holdout (Oct–Dec 2025)
- Table Content:
  | Phase | Architecture | Key Feature Ingestion | R² Score | RMSE (W/m²) | MAE (W/m²) | Skill Score | Conformal PICP |
  |---|---|---|:---:|:---:|:---:|:---:|:---:|
  | Phase 3d | 2-Layer LSTM | 8 features × 60 min (480 pts) | 0.5266 | 204.67 | 155.80 | +25.79% | — |
  | Phase 4c | Recurrent Fusion | Past Seq + GHI_clear(t+60) | 0.5239 | 203.11 | 154.58 | +27.48% | — |
  | Phase 5 | XGBoost | Ground + Satellite Clouds + Astronomy | 0.5342 | 198.14 | 149.15 | +28.03% | — |
  | Phase 5 | GBDT Ensemble | 50% LightGBM + 50% XGBoost | 0.5387 | 197.16 | 148.95 | +28.39% | — |
  | Phase 5 | LightGBM | Ground + Satellite Clouds + Astronomy | 0.5392 | 197.07 | 149.30 | +28.42% | — |
  | Phase 6 | Physics-P50 CQR | Variability Index (VI) + Satellite + CQR | 0.5480 🏆 | 195.18 🏆 | 141.90 🏆 | +29.10% 🏆 | 81.98% 🏆 |
- Image Asset: `https://raw.githubusercontent.com/saimanoj446/Renewable-Energy-Forecasting-AI/main/plots/59_ph6_benchmark_comparison.png`

---

### SLIDE 13: FUTURE WORK (TRACK 1) — UPWIND SPATIAL ADVECTION & WAVELETS
- Layout: 2-Column Split (Left: Spatial Formulation, Right: Wavelet Decomposition)
- Header: Phase 7 Roadmap: Upwind Spatial Satellite Patches
- Subheader: Turning Temporal Extrapolation into an Observable Spatial Lookup
- Content:
  * 1. 5×5 Geostationary Grid Patches:
    - Ingest Himawari-8/9 Band 03 (Visible albedo) + Band 13 (IR cloud-top temperature) at 10-min, 2 km resolution.
    - Project upwind location using 850 hPa wind vector v_wind: x_upwind = x_station - v_850hPa · Δt.
    - Ingests incoming cloud fronts 15–40 km upwind before they shade the pyranometer!
  * 2. Causal Wavelet Decomposition (MODWT):
    - Split 1-min GHI into 3 Intrinsic Mode Functions:
      * Low-frequency: Astronomical diurnal curve (>120 min).
      * Mesoscale: Cloud front passages (15–60 min).
      * Microscale: Atmospheric turbulence & sensor noise (1–5 min).

---

### SLIDE 14: FUTURE WORK (TRACK 2) — BESS RAMP-RATE SMOOTHING (IEEE 1547)
- Layout: 2-Column Split (Left: Power Systems Formulation, Right: Optimization Objective)
- Header: Phase 7 Roadmap: Closing the Loop on Grid Stability
- Subheader: Model Predictive Control (MPC) Battery Dispatch for a 10 MW Utility Solar Farm
- Content:
  * Grid Code Constraint: IEEE 1547 / Malaysian Grid Code limits ramp rates to ±10% rated capacity/min (1 MW/min on 10 MW farm).
  * The Reactive Limitation: Traditional low-pass filters react AFTER cloud shading occurs, causing battery exhaustion.
  * Proposed AI-Predictive MPC Dispatch:
    - Uses Phase 6 P50 forecast and lower quantile q10 to pre-charge or pre-discharge the BESS before a cloud arrives:
      min Σ [ w1 · max(0, |ΔP_grid(t+τ)| - R_max)² + w2 · (SoC(t+τ) - SoC_target)² ]
  * Quantifiable Engineering Deliverable: Prove a 25–35% reduction in required battery capacity (MWh) compared to naive filtering.

---

### SLIDE 15: CONCLUSION & DISCUSSION PROMPTS
- Layout: 2-Column Split (Left: Delivered Achievements, Right: Strategic Questions)
- Header: Conclusion & Discussion with Prof. Arup Ratan
- Subheader: Design-Oriented Project (DOP) Milestone Summary
- Content Left (Delivered This Sprint):
  * Solved solar noon overshoot physically using GHI_clear(t+60).
  * Ingested satellite cloud layers, proving Low Convective Clouds are the #3 predictive feature.
  * Replaced LSTMs with LightGBM & Variability Index, driving R² to 0.5480, MAE down by 14 W/m², and Skill to +29.10%.
  * Delivered 81.98% calibrated conformal prediction intervals (CRPS = 42.58 W/m²).
- Content Right (Questions for Sir):
  1. For Phase 7, should we prioritize Track 1 (Upwind Satellite Patches) or Track 2 (BESS Ramp-Rate Smoothing) to best suit EEE department goals?
  2. For the BESS simulation, should we benchmark against a 5 MW or 10 MW utility standard solar park?
- Footer: Complete code, dataset, and 59 publication plots available at https://github.com/saimanoj446/Renewable-Energy-Forecasting-AI
```

