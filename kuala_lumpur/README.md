# Kuala Lumpur 1-Minute Solar Forecasting Dataset (2025)

**Project:** DOP — Forecasting of Renewable Energy Resources Using AI  
**Institution:** BITS Pilani, Hyderabad Campus | Department of EEE  
**Supervisor:** Prof. Arup Ratan  

---

## 📁 Directory Structure

```
d:/DOP/kuala_lumpur/
├── 2025/                                # Raw daily CSV files (365 files, Jan-Dec 2025)
│   ├── 2025-01/                         # Daily 1-min GHI readings (07:00 - 18:59)
│   ├── ...
│   └── 2025-12/
├── kl_hourly_weather.csv               # Downloaded Open-Meteo hourly weather (cached)
├── kuala_lumpur_1min.csv                # Merged 1-minute dataset (262,020 rows, 29 cols)
├── 06_prepare_1min_dataset.py          # Phase 3b: Data ingestion, interpolation & feature engineering
├── 07_minute_eda.py                    # Phase 3c: 1-min EDA, 4-method feature importance proof
├── 08_lstm_forecast.py                 # Phase 3d: PyTorch 2-layer LSTM sequence forecast & MPP power
├── 09_phase4_clearsky_lstm.py          # Phase 4a: Clear-Sky Index k_t + Conv1D+BiLSTM+Attention
├── 10_phase4b_ghi_clearsky_lstm.py     # Phase 4b: GHI + GHI_clear residual LSTM
├── 11_phase4c_t60_specialist.py        # Phase 4c: Dedicated t+60 Specialist + Future Astronomy
├── 12_phase5_satellite_lightgbm.py     # Phase 5: Satellite Cloud Layers + LightGBM/XGBoost Regressors
├── phase3c_kl_feature_results.txt      # Numerical comparison: full (25) vs reduced (8) features
├── lstm_results.txt                    # Phase 3d LSTM multi-horizon & MPP metrics
├── phase4_results.txt                  # Phase 4a k_t LSTM results
├── phase4b_results.txt                 # Phase 4b GHI+GHI_clear LSTM results
├── phase4c_results.txt                 # Phase 4c Dedicated t+60 results
├── phase5_results.txt                  # Phase 5 Satellite-Enhanced LightGBM & XGBoost results
├── plots/                              # Generated figures (plots 29–55)
└── README.md                           # This documentation file
```

---

## 🌍 Dataset Overview

- **Location:** Kuala Lumpur, Malaysia (Latitude: `3.1390° N`, Longitude: `101.6869° E`)
- **Climate Zone:** Tropical rainforest (Af), equatorial solar irradiance profiles with frequent convective cloud transients
- **Temporal Resolution:** 1 minute (720 readings/day, 07:00:00 to 18:59:00)
- **Timeframe:** Full Year 2025 (365 days, 262,020 minute records)
- **Primary Source:** Published in *Data in Brief* (measured solar irradiance sensor station)
- **Auxiliary Weather & Satellite:** Open-Meteo Historical Archive API (Temperature, Relative Humidity, Wind, Pressure, DNI, DHI, Total Cloud Cover, Low Convective Cloud Cover)

---

## 🎯 Phase Summary & Research Findings

### Phase 3d: PyTorch LSTM Baseline (The "480 Data Points" Rule)
Validated that an input sequence of **8 physical features × 60 minutes = 480 data points** captures temporal dependencies:
- **Tensor Input:** `(batch_size, 60, 8)`
- **+1m Forecast:** $R^2 = 0.8978$, $\text{RMSE} = 91.22\ W/m^2$ (+3.07% Skill)
- **+60m Forecast:** $R^2 = 0.5266$, $\text{RMSE} = 204.67\ W/m^2$ (+25.79% Skill)

### Phase 4: Diurnal Curve & Solar Noon Analysis
Investigated the $t+60$ forecasting problem around solar noon (12:00 PM). With raw GHI, linear trend extrapolation overshoots when approaching the peak.
- **Phase 4a:** Clear-sky index ($k_t = GHI / GHI_{\text{clear}}$) model with Conv1D + BiLSTM + Attention.
- **Phase 4b:** GHI residual model incorporating $GHI_{\text{clear}}$ as an exogenous feature. Stabilized validation loss (converged at epoch 11).
- **Phase 4c:** Dedicated single-horizon model for $t+60$ using **Known Future Astronomy $GHI_{\text{clear}}(t+60)$**. Enforced strict physical bounds with 0% runaway overshoot past solar noon (+27.48% Skill).

### Phase 5: Satellite-Enhanced Machine Learning (Beyond LSTM)
Replaced the recurrent sequence formulation with **Gradient Boosted Decision Trees (LightGBM & XGBoost)** and ingested **Geostationary Satellite Cloud Cover** via API:
- Ingested satellite cloud layers: `Cloud_Cover`, `Cloud_Cover_Low`, `Cloud_Cover_Mid`, `Cloud_Cover_High`.
- Feature importance confirmed **`Cloud_Cover_Low` is the #3 most important feature** in predicting 60-minute solar irradiance.
- Achieved the **project's top $t+60$ performance**:
  - **LightGBM $R^2 = 0.5392$** (highest of all phases)
  - **RMSE = $197.07\ W/m^2$** (reduced by $7.6\ W/m^2$ vs LSTM)
  - **MAE = $149.30\ W/m^2$** (reduced by $6.5\ W/m^2$ vs LSTM)
  - **IEEE Skill Score = $+28.42\%$** over persistence

---

## 📊 Comprehensive Model Benchmark (+60 Min Horizon)

Evaluated strictly on the out-of-time test set (Oct–Dec 2025):

| Phase | Model Architecture | Features Used | $R^2$ Score | RMSE ($W/m^2$) | MAE ($W/m^2$) | Skill Score (%) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Phase 3** | 2-Layer PyTorch LSTM | 8 features $\times$ 60 min (480 pts) | 0.5266 | 204.67 | 155.80 | +25.79% |
| **Phase 4c** | Dedicated Recurrent Fusion | Past Seq + Known Future $GHI_{\text{clear}}(t+60)$ | 0.5239 | 203.11 | 154.58 | +27.48% |
| **Phase 5** | **LightGBM Regressor** | **Ground + Satellite Clouds + Astronomy** | **0.5392** 🏆 | **197.07** 🏆 | **149.30** | **+28.42%** 🏆 |
| **Phase 5** | **XGBoost Regressor** | **Ground + Satellite Clouds + Astronomy** | **0.5342** | **198.14** | **149.15** 🏆 | **+28.03%** |
| **Phase 5** | **GBDT Ensemble** | **50% LightGBM + 50% XGBoost** | **0.5387** | **197.16** | **148.95** 🏆 | **+28.39%** |

---

## 🚀 Execution Instructions

Run from anywhere in the terminal or on Google Colab:

```powershell
# 1. Dataset Preparation (Ground Pyranometer + Open-Meteo):
python d:\DOP\kuala_lumpur\06_prepare_1min_dataset.py

# 2. 1-Minute EDA & Feature Selection Proof:
python d:\DOP\kuala_lumpur\07_minute_eda.py

# 3. Phase 3: PyTorch LSTM Sequence Model:
python d:\DOP\kuala_lumpur\08_lstm_forecast.py

# 4. Phase 4c: Dedicated t+60 Forecaster with Future Astronomy:
python d:\DOP\kuala_lumpur\11_phase4c_t60_specialist.py

# 5. Phase 5: Satellite-Enhanced LightGBM & XGBoost Models:
python d:\DOP\kuala_lumpur\12_phase5_satellite_lightgbm.py
```
