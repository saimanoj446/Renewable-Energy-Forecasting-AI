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
├── 06_prepare_1min_dataset.py          # Data ingestion, interpolation & feature engineering
├── 07_minute_eda.py                    # 1-min EDA, 4-method feature importance proof
├── 08_lstm_forecast.py                 # PyTorch 2-layer LSTM sequence forecast & MPP power
├── phase3c_kl_feature_results.txt      # Numerical comparison: full (25) vs reduced (8) features
├── lstm_results.txt                    # Detailed PyTorch LSTM multi-horizon & MPP metrics
├── plots/                              # Generated figures (plots 29–38)
│   ├── 29_kl_eda_overview.png
│   ├── 30_kl_correlation_heatmap.png
│   ├── 31_kl_1min_timeseries_week.png
│   ├── 32_kl_feature_importance.png
│   ├── 33_kl_480_datapoints_matrix.png
│   ├── 34_kl_lstm_training_loss.png
│   ├── 35_kl_lstm_scatter_and_residuals.png
│   ├── 36_kl_lstm_1min_walkforward_day.png
│   ├── 37_kl_lstm_mpp_power.png
│   └── 38_kl_lstm_multi_horizon.png
└── README.md                           # This documentation file
```

---

## 🌍 Dataset Overview

- **Location:** Kuala Lumpur, Malaysia (Latitude: `3.1390° N`, Longitude: `101.6869° E`)
- **Climate Zone:** Tropical rainforest (Af), equatorial solar irradiance profiles with frequent convective cloud transients
- **Temporal Resolution:** 1 minute (720 readings/day, 07:00:00 to 18:59:00)
- **Timeframe:** Full Year 2025 (365 days, 262,020 minute records)
- **Primary Source:** Published in *Data in Brief* (measured solar irradiance sensor station)
- **Auxiliary Weather:** Open-Meteo Historical Archive API (temperature, relative humidity, wind speed, wind direction, surface pressure, DNI, DHI) interpolated to 1-minute intervals

---

## 🎯 Purpose & Professor Meeting Proof

This sub-dataset fulfills the specific directive from **Prof. Arup Ratan**:
1. **Minute-Level Proof:** Prove whether the same feature selection conclusions held at hourly resolution (Rajasthan) remain true at **1-minute resolution**.
2. **The "480 Data Points" Rule:** Validate that an input window of **8 key features × 60 minutes = 480 data points** captures virtually 100% of the predictive information:
   - **Full Model (25 features):** $R^2 = 0.9162$
   - **Reduced Model (Top 8 features):** $R^2 = 0.9169$ (Drop $\approx 0.0000$)
3. **Deep Learning (PyTorch LSTM) Implementation:**
   - Input sequence tensor: `(batch_size, 60, 8)` representing 60 time steps × 8 physical features = **480 input elements**.
   - Out-of-time evaluation on Oct–Dec 2025 (55,292 testing sequence windows):

| Horizon | Lead Time | $R^2$ Score | RMSE ($W/m^2$) | MAE ($W/m^2$) | Skill Score vs Persistence |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **+1 min** | 1 minute | **0.8978** | **91.22** | **47.77** | **+3.07%** |
| **+5 min** | 5 minutes | **0.7494** | **142.99** | **89.99** | **+11.66%** |
| **+15 min** | 15 minutes | **0.6321** | **173.90** | **121.30** | **+17.16%** |
| **+30 min** | 30 minutes | **0.5712** | **189.50** | **139.03** | **+20.30%** |
| **+60 min** | 60 minutes | **0.5266** | **204.67** | **155.80** | **+25.79%** |

4. **Physics-Based PV Maximum Power Point (MPP):**
   - $P_{mpp} = P_{stc} \times \frac{GHI}{1000} \times [1 + \alpha (T_{cell} - 25)]$
   - Module: 300W STC ($\alpha = -0.0035/^\circ\text{C}$, $\text{NOCT} = 45^\circ\text{C}$)
   - **Daytime $R^2 = 0.9020$** | **MAE = 12.41 W / module** | **RMSE = 23.10 W / module**

---

## 🚀 Execution Instructions

Run from anywhere in the terminal (paths auto-resolve):

```powershell
# 1. Rebuild or verify the merged 1-min dataset:
python d:\DOP\kuala_lumpur\06_prepare_1min_dataset.py

# 2. Run EDA, feature importance proof, and generate plots 29-33:
python d:\DOP\kuala_lumpur\07_minute_eda.py

# 3. Train & evaluate PyTorch LSTM, generate plots 34-38:
python d:\DOP\kuala_lumpur\08_lstm_forecast.py
```
