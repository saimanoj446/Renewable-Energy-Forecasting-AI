# Renewable-Energy-Forecasting-AI

Physics-informed and machine learning models for short-term solar and wind power forecasting, integrated with grid stability analysis.

**Current Phase:** Phase 9 ✅ — SAPEC-Grade 2D Parametric Horizon Sweep & Multi-Model Forecasting (Plots 74–78)

**Design-Oriented Project (DOP)** | Electrical & Electronics Engineering Department | **BITS Pilani, Hyderabad Campus**  
**Under the Guidance of:** Prof. Arup Ratan

---

## 📌 Project Overview & Roadmap

Accurate forecasting of **Global Horizontal Irradiance (GHI)** is essential for modern grid scheduling, ramp-rate mitigation, and minimizing renewable curtailment.

This repository implements an end-to-end multi-phase AI forecasting framework across two distinct climate regimes:

- **Phase 1:** Historical Data Preprocessing, Exploratory Data Analysis (EDA), Cyclical Feature Engineering, and Random Forest Baseline Regression on 5 years of hourly solar data (Rajasthan).
- **Phase 2:** Multi-model benchmarking (Random Forest vs XGBoost vs Gradient Boosting vs SVR) with TimeSeriesSplit validation.
- **Phase 3:** 
  - **Phase 3a/b:** Feature selection proof on Rajasthan hourly dataset.
  - **Phase 3c:** Ingestion of 1-minute ground pyranometer data (Kuala Lumpur 2025, 262,020 records) + Open-Meteo weather API interpolation.
  - **Phase 3d:** Validation of Prof. Arup Ratan's **"8 features × 60 minutes = 480 data points"** formulation using a PyTorch 2-layer LSTM sequence model across multi-horizons (+1m, +5m, +15m, +30m, +60m) and Sandia/IEC photovoltaic MPP power conversion.
- **Phase 4:** Diurnal Solar Geometry & Noon Peak Analysis ($GHI_{\text{clear}}$ normalization, recurrent fusion with future astronomy to eliminate linear trend overshoot past 12:00 PM).
- **Phase 5:** Ingestion of Geostationary Satellite Cloud Cover (Low/Mid/High layers) via Open-Meteo Satellite API, and deployment of **LightGBM & XGBoost Gradient Boosted Trees** achieving top $t+60$ forecast accuracy ($R^2 = 0.5392$, $\text{Skill} = +28.42\%$).
- **Phase 6:** Conformalized Quantile Regression (CQR) & Arc-Length Variability Index ($VI_{30}$), establishing an $81.98\%$ calibrated prediction interval and $R^2 = 0.5480$ 🏆.
- **Phase 7:** 3D Cloud Parallax & Ray-Tracing Shadow Model ($\vec{d}_{\text{shadow}} = H_c \tan\theta_s$, $\vec{d}_{\text{parallax}} = H_c \tan\theta_v$, net offset $D_{\text{net}}$), proving `dx_net_offset` is the #3 most influential feature.
- **Phase 7b:** 2D Parametric Horizon & Lookback Optimization ($36\text{ pairs}$ across $x \in [15, 120]\text{m}, t_d \in [5, 60]\text{m}$), proving $x = 60\text{m}$ is the universal sweet spot and parallax produces unanimously positive gains at $t_d \ge 30\text{m}$.
- **Phase 7c:** Breaking the "Double Penalty" Ceiling via Target Aggregation (1-min vs 5-min CAISO RTD vs 15-min Dispatch) and causal online EWMA residual filtering ($b_t = 0.85 b_{t-1} + 0.15 e_t$), achieving **$R^2 = 0.7164$** (5-min) and **$R^2 = 0.7633$** (15-min).
- **Phase 8:** Physics-Informed PV Array Power Conversion (Sandia thermal lag $\tau = 7\text{ min}$, $1.45\text{ MW}$ midday derating) & Predictive BESS Grid Stability Dispatch, proving **$99.8\%$ IEEE 1547 ramp compliance** and a **$32.7\%$ battery capacity sizing reduction**.
- **Phase 9:** SAPEC 2027 Model Optimization & 2D ($x, t_d$) Global Grid Sweep ($x \in [15, 120]\text{m}, t_d \in [5, 60]\text{m}$). Establishes peak operational accuracy of **$R^2 = 0.8553$** ($\text{RMSE} = 106.59\text{ W/m}^2$, $\text{Skill} = +20.44\%$) at $t_d = 5\text{ min}$ (CAISO Real-Time Dispatch / Microgrid BESS timescale) with $x^* = 60\text{ min}$, and maps the complete operational Pareto frontier across all 36 combinations for power electronics inverter control.

---

## 📊 Summary of Benchmark Results

### 1. Rajasthan Dataset (Hourly Resolution, 2010–2014)

| Model | R² Score | MAE (W/m²) | RMSE (W/m²) | nRMSE (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest** | **0.9846** 🏆 | **12.75** | **39.42** | **3.96%** |
| **Gradient Boosting** | 0.9841 | 14.90 | 40.05 | 4.02% |
| **XGBoost** | 0.9842 | 13.88 | 39.92 | 4.01% |
| **SVR (RBF Kernel)** | 0.9839 | 16.15 | 40.28 | 4.04% |

---

### 2. Kuala Lumpur Dataset (1-Minute Resolution, Full Year 2025)

#### Multi-Horizon PyTorch LSTM Sequence Model (Phase 3d)
Input Tensor: `(batch_size, 60, 8)` = **480 data points per sample** (evaluated on Oct–Dec 2025 holdout):

| Horizon | Lead Time | $R^2$ Score | RMSE ($W/m^2$) | MAE ($W/m^2$) | IEEE Skill Score vs Persistence |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **+1 min** | 1 minute | **0.8978** | **91.22** | **47.77** | **+3.07%** |
| **+5 min** | 5 minutes | **0.7494** | **142.99** | **89.99** | **+11.66%** |
| **+15 min** | 15 minutes | **0.6321** | **173.90** | **121.30** | **+17.16%** |
| **+30 min** | 30 minutes | **0.5712** | **189.50** | **139.03** | **+20.30%** |
| **+60 min** | 60 minutes | **0.5266** | **204.67** | **155.80** | **+25.79%** |

#### Dedicated +60 Minute Horizon Benchmark (Phase 3 vs Phase 4c vs Phase 5)

| Phase | Model Architecture | Features / Data Sources | $R^2$ Score | RMSE ($W/m^2$) | MAE ($W/m^2$) | Skill Score (%) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Phase 3** | 2-Layer PyTorch LSTM | 8 features $\times$ 60 min (480 pts) | 0.5266 | 204.67 | 155.80 | +25.79% |
| **Phase 4c** | Recurrent Fusion | Past Seq + Known Future Astronomy $GHI_{\text{clear}}(t+60)$ | 0.5239 | 203.11 | 154.58 | +27.48% |
| **Phase 5** | **LightGBM Regressor** | **Ground Pyranometer + Satellite Clouds + Astronomy** | **0.5392** | **197.07** | **149.30** | **+28.42%** |
| **Phase 5** | **XGBoost Regressor** | **Ground Pyranometer + Satellite Clouds + Astronomy** | **0.5342** | **198.14** | **149.15** | **+28.03%** |
| **Phase 5** | **GBDT Ensemble** | **50% LightGBM + 50% XGBoost** | **0.5387** | **197.16** | **148.95** | **+28.39%** |
| **Phase 6** | **Physics-P50 Conformal Hybrid** | **Variability Index (VI) + Satellite + Conformal CQR** | **0.5480** 🏆 | **195.18** 🏆 | **141.90** 🏆 | **+29.10%** 🏆 |
| *Phase 6 (Probabilistic)* | *Conformal 80% Prediction Interval* | *Calibrated Coverage (PICP) = **81.98%** (Nominal 80%)* | *PINAW: 33.7%* | *CRPS: 42.58 W/m²* | *Pinball Loss: 42.5 W/m²* | *IEEE Standard ✅* |
| **Phase 7** | 3D Ray-Tracing Parallax | Ground Pyranometer + Damped Parallax Vector ($D_{\text{net}}$) | 0.5423 | 196.40 | 142.10 | +28.90% |
| **Phase 7b** | 2D Parametric Sweet Spot | Optimal Lookback $x=60\text{m}$ across Horizons $t_d \in [5, 60]\text{m}$ | 0.7903 (5m) $\to$ 0.5423 (60m) | — | — | Parallax gain $\forall t_d \ge 30\text{m}$ |
| **Phase 7c** | **Anti-Double Penalty + EWMA** | **5-Min CAISO RTD Target Block ($t+60\text{m}$)** | **0.7164 🏆** | **149.30 🏆** | **111.93 🏆** | **+36.62% 🏆** |
| **Phase 7c** | **Anti-Double Penalty + EWMA** | **15-Min Market Dispatch Block ($t+60\text{m}$)** | **0.7633 🏆** | **131.31 🏆** | **99.24 🏆** | **+41.14% 🏆** |
| **Phase 8** | **Dynamic PV AC Power** | **10 MWp DC / 8 MW AC Farm (Sandia Thermal Lag $\tau=7\text{m}$)** | **0.7248 (5m)** / **0.7680 (15m)** | **0.88 MW** | **0.51 MW** | **14.5% Midday Derating Loss** |
| **Phase 8** | **Predictive BESS Grid Dispatch** | **IEEE 1547 Ramp Smoothing ($\pm 0.80\text{ MW/min}$ Limit)** | **> 99.8% Compliance 🏆** | **0.18% Viol** | — | **32.7% BESS Sizing Reduction** |
| **Phase 9** | **SAPEC Peak Operational (t+5m)** | **Global Optimum ($x^*=60\text{m}, t_d^*=5\text{m}$) — RTD BESS Control** | **0.8553 🏆** | **106.59 🏆** | **64.09 🏆** | **+20.44% (Peak R²)** |
| **Phase 9** | **SAPEC Intra-Hour (t+10m)** | **$x^*=90\text{m}, t_d=10\text{m}$ — Active Power Ramp Mitigator** | **0.7960** | **126.44** | **82.03** | **+24.13%** |
| **Phase 9** | **SAPEC Intra-Hour (t+15m)** | **$x^*=90\text{m}, t_d=15\text{m}$ — CAISO 15-Min Market Dispatch** | **0.7587** | **137.39** | **93.96** | **+25.72%** |
| **Phase 9** | **SAPEC Horizon Sweep (t+30m)** | **$x^*=90\text{m}, t_d=30\text{m}$ — Sub-Hourly Firm Capacity** | **0.7040** | **151.95** | **110.45** | **+27.62%** |

---

## 📈 Key Visualizations & Plots

All generated publication plots are stored in [`plots/`](plots/):

- **Plots 01–28:** Rajasthan Exploratory Data Analysis, Model Benchmarks, and Feature Selection.
- **Plots 29–33:** Kuala Lumpur 1-Minute EDA, Pearson Heatmap, and 480-Datapoint Proof.
- **Plots 34–38:** PyTorch LSTM Sequence Convergence, Walk-Forward Day, Photovoltaic MPP Power Conversion, and Multi-Horizon Skill Score Curve.
- **Plots 45–48:** Phase 4b Residual LSTM and Solar Noon Diurnal Analysis.
- **Plots 49–51:** Phase 4c Dedicated $t+60$ Fusion with Astronomical Ceiling Bounds.
- **Plots 52–55:** Phase 5 Satellite Cloud Feature Importance, GBDT Model Comparisons, and Continuous Walk-Forward Tracking.
- **Plots 56–59:** Phase 6 Conformalized Quantile Fan Charts, Calibration Diagrams, and Physics Feature Importance.
- **Plots 60–62:** Phase 7b 2D Parametric Lookback vs Horizon Optimization Heatmaps and Pareto Curves.
- **Plots 63–65:** Phase 7 3D Cloud Parallax & Ray-Tracing Geometry, Feature Importance, and Walk-Forward Profiles.
- **Plots 66–67:** Phase 7b Parallax Horizon Benefit Heatmaps and Frontiers.
- **Plots 68–70:** Phase 7c Target Aggregation Breakthrough, Walk-Forward Head-to-Head Comparisons, and Online EWMA Filter Impact.
- **Plots 71–73:** Phase 8 Sandia Module Thermal Derating, IEEE 1547 BESS Convective Storm Smoothing, and Sizing Pareto Frontier.
- **Plots 74–78:** Phase 9 SAPEC Benchmark Comparison, 2D Parametric Sweep Heatmaps, Pareto Frontier across Lookbacks, Multi-Horizon Walk-Forward Tracking, and Enriched Feature Importance.
- **Plots 79–81:** 3D Parametric Response Surfaces ($R^2$ vs $x$ and $t_d$, $\text{MSE}$ vs $x$ and $t_d$, and Dual-Panel Coupled Manifold) requested by Prof. Arup Ratan.

---

## 🗂️ Repository Structure

```
├── 01_eda.py                         # Phase 1: Data cleaning, cyclical feature engineering & EDA
├── 02_model.py                       # Phase 1: Random Forest regressor & baseline evaluations
├── 03_model_comparison.py            # Phase 2: RF vs XGBoost vs GBM vs SVR comparison
├── 04_key_features.py                # Phase 3a: 4-method feature importance on hourly data
├── 05_rolling_window_forecast.py     # Phase 3b: Rolling walk-forward hourly evaluation
├── kuala_lumpur/                     # Phase 3 to 8: 1-Minute Tropical Climate Research
│   ├── 06_prepare_1min_dataset.py    # Merges 365 daily GHI files + Open-Meteo weather
│   ├── 07_minute_eda.py              # 1-min EDA and 480-point feature reduction proof
│   ├── 08_lstm_forecast.py           # Phase 3d: PyTorch 2-layer LSTM sequence model
│   ├── 09_phase4_clearsky_lstm.py    # Phase 4a: Clear-sky index kt BiLSTM model
│   ├── 10_phase4b_ghi_clearsky_lstm.py # Phase 4b: GHI + GHI_clear residual model
│   ├── 11_phase4c_t60_specialist.py  # Phase 4c: Dedicated t+60 specialist model
│   ├── 12_phase5_satellite_lightgbm.py # Phase 5: Satellite cloud layers + LightGBM/XGBoost
│   ├── 13_phase6_probabilistic_conformal.py # Phase 6: Conformalized Quantile Regression (CQR)
│   ├── 14_lookback_horizon_optimization.py  # Phase 7b: 36-cell parametric grid sweep
│   ├── 15_phase7_parallax_cloud_model.py    # Phase 7: 3D Cloud Parallax & Ray-Tracing
│   ├── 16_parallax_lookback_horizon_optimization.py # Phase 7b: Parallax parametric mapping
│   ├── 17_phase7c_anti_double_penalty_model.py      # Phase 7c: Target aggregation & online EWMA
│   ├── 18_phase8_pv_power_grid_balance.py           # Phase 8: PV Thermal power + BESS dispatch
│   ├── kuala_lumpur_1min.csv         # Merged 1-min ground dataset (262,020 records)
│   ├── kl_hourly_weather.csv         # Open-Meteo historical weather archive
│   ├── plots/                        # Kuala Lumpur specific figures (plots 29–73)
│   └── README.md                     # Kuala Lumpur sub-study documentation
├── plots/                            # Master figure directory (plots 01–73)
├── presentation.html                 # Interactive browser presentation deck
├── ppt_content.md                    # Slide-by-slide markdown script & speaker notes
├── SLIDE_CONTEXT.md                  # Comprehensive slide generation specifications & raw links
├── Solar_Forecasting_Prof_Meeting.pptx # PowerPoint presentation deck
├── dop_project_guide.md              # Research references & mathematical methodology
├── midsem_report.md                  # Comprehensive mid-semester project report
└── README.md                         # Main project documentation
```

---

## 🚀 How to Run

### Google Colab / Local Execution
```python
# 1. Pull the latest pipeline from GitHub:
!git pull origin main

# 2. Run Phase 7c Breakthrough Model (R² = 0.7164 on 5-min RTD, R² = 0.7633 on 15-min Dispatch):
%run kuala_lumpur/17_phase7c_anti_double_penalty_model.py

# 3. Run Phase 8 Dynamic PV Power & BESS Grid-Balancing Simulation (Plots 71, 72, 73):
%run kuala_lumpur/18_phase8_pv_power_grid_balance.py
```

