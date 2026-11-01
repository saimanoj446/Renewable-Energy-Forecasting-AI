# Renewable-Energy-Forecasting-AI

Physics-informed and machine learning models for short-term solar and wind power forecasting, integrated with grid stability analysis.

**Current Phase:** Phase 5 ✅ — Satellite-Enhanced Machine Learning & Multi-Horizon Deep Learning Benchmarking

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
| **Phase 5** | **LightGBM Regressor** | **Ground Pyranometer + Satellite Clouds + Astronomy** | **0.5392** 🏆 | **197.07** 🏆 | **149.30** | **+28.42%** 🏆 |
| **Phase 5** | **XGBoost Regressor** | **Ground Pyranometer + Satellite Clouds + Astronomy** | **0.5342** | **198.14** | **149.15** 🏆 | **+28.03%** |
| **Phase 5** | **GBDT Ensemble** | **50% LightGBM + 50% XGBoost** | **0.5387** | **197.16** | **148.95** 🏆 | **+28.39%** |

---

## 📈 Key Visualizations & Plots

All generated publication plots are stored in [`plots/`](plots/):

- **Plots 01–28:** Rajasthan Exploratory Data Analysis, Model Benchmarks, and Feature Selection.
- **Plots 29–33:** Kuala Lumpur 1-Minute EDA, Pearson Heatmap, and 480-Datapoint Proof.
- **Plots 34–38:** PyTorch LSTM Sequence Convergence, Walk-Forward Day, Photovoltaic MPP Power Conversion, and Multi-Horizon Skill Score Curve.
- **Plots 45–48:** Phase 4b Residual LSTM and Solar Noon Diurnal Analysis.
- **Plots 49–51:** Phase 4c Dedicated $t+60$ Fusion with Astronomical Ceiling Bounds.
- **Plots 52–55:** Phase 5 Satellite Cloud Feature Importance, GBDT Model Comparisons, and Continuous Walk-Forward Tracking.

---

## 🗂️ Repository Structure

```
├── 01_eda.py                         # Phase 1: Data cleaning, cyclical feature engineering & EDA
├── 02_model.py                       # Phase 1: Random Forest regressor & baseline evaluations
├── 03_model_comparison.py            # Phase 2: RF vs XGBoost vs GBM vs SVR comparison
├── 04_key_features.py                # Phase 3a: 4-method feature importance on hourly data
├── 05_rolling_window_forecast.py     # Phase 3b: Rolling walk-forward hourly evaluation
├── kuala_lumpur/                     # Phase 3, 4, 5: 1-Minute Tropical Climate Research
│   ├── 06_prepare_1min_dataset.py    # Merges 365 daily GHI files + Open-Meteo weather
│   ├── 07_minute_eda.py              # 1-min EDA and 480-point feature reduction proof
│   ├── 08_lstm_forecast.py           # Phase 3d: PyTorch 2-layer LSTM sequence model
│   ├── 09_phase4_clearsky_lstm.py    # Phase 4a: Clear-sky index kt BiLSTM model
│   ├── 10_phase4b_ghi_clearsky_lstm.py # Phase 4b: GHI + GHI_clear residual model
│   ├── 11_phase4c_t60_specialist.py  # Phase 4c: Dedicated t+60 specialist model
│   ├── 12_phase5_satellite_lightgbm.py # Phase 5: Satellite cloud layers + LightGBM/XGBoost
│   ├── kuala_lumpur_1min.csv         # Merged 1-min ground dataset (262,020 records)
│   ├── kl_hourly_weather.csv         # Open-Meteo historical weather archive
│   ├── plots/                        # Kuala Lumpur specific figures (plots 29–55)
│   └── README.md                     # Kuala Lumpur sub-study documentation
├── plots/                            # Master figure directory (plots 01–55)
├── presentation.html                 # Interactive browser presentation deck
├── ppt_content.md                    # Slide-by-slide markdown script & speaker notes
├── Solar_Forecasting_Prof_Meeting.pptx # PowerPoint presentation deck
├── dop_project_guide.md              # Research references & mathematical methodology
├── midsem_report.md                  # Comprehensive mid-semester project report
└── README.md                         # Main project documentation
```

---

## 🚀 How to Run

### Local Execution (PowerShell / Command Prompt)
```powershell
# Run the 1-minute Kuala Lumpur LSTM baseline:
python kuala_lumpur/08_lstm_forecast.py

# Run the Phase 5 Satellite-Enhanced LightGBM model:
python kuala_lumpur/12_phase5_satellite_lightgbm.py
```
