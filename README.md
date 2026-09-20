# Renewable-Energy-Forecasting-AI

Physics-informed and machine learning models for short-term solar and wind power forecasting, integrated with grid stability analysis.

**Current Phase:** Phase 2 ✅ — Model Comparison Study (RF vs XGBoost vs Gradient Boosting vs SVR)

**Design-Oriented Project (DOP)** | Electrical & Electronics Engineering Department | **BITS Pilani, Hyderabad Campus**  
**Under the Guidance of:** Prof. Arup Ratan

---

## 📌 Project Overview

Solar energy is intermittent and weather-dependent. Accurate forecasting of **Global Horizontal Irradiance (GHI)** is essential for grid scheduling, unit commitment, and minimizing curtailment losses.

This repository implements a multi-phase AI forecasting framework:
- **Phase 1 (Current):** Historical Data Preprocessing, Exploratory Data Analysis (EDA), Cyclical Feature Engineering, and Random Forest Baseline Regression on 5 years of hourly solar data.
- **Phase 2:** Granular hourly forecasting, model benchmarking against Gradient Boosting, XGBoost, and SVR.
- **Phase 3:** Real-time ingestion via meteorological APIs (NASA POWER / OpenWeatherMap) for campus-specific forecasting (Hyderabad).
- **Phase 4:** Deep learning models (BiLSTM, Temporal Transformers, PINNs) & interactive deployment dashboard.

---

## 📊 Phase 1 Baseline Results (Random Forest)

Trained on 4 years (2010–2013, 35,040 samples) and evaluated on a holdout test year (2014, 8,760 samples):

| Metric | Training Set (2010–2013) | Test Set (2014) |
| :--- | :--- | :--- |
| **R² Score** | **0.9830** | **0.9695** |
| **MAE (Mean Absolute Error)** | 14.15 W/m² | **21.00 W/m²** |
| **RMSE (Root Mean Squared Error)** | 39.81 W/m² | **55.14 W/m²** |
| **nRMSE (Normalized RMSE)** | 4.02% | **5.56%** |
| **5-Fold Cross-Validation R²** | — | **0.9652 ± 0.0078** |

*Literature baseline (reference video on Random Forest solar forecasting) reported ~28% nRMSE for 1-hour ahead horizons; our model achieves 5.56% on the test set.*

---

## 📊 Phase 2 Results — Model Comparison Study

**Phase 2 Additions:** 6 new lag/rolling features (GHI_lag1/2/3, GHI_roll3h, GHI_roll6h, Temp_lag1), bringing total features from 13 → 19. All models tuned with `RandomizedSearchCV` and evaluated with `TimeSeriesSplit` (5-fold).

> Full numbers are in [`phase2_results.txt`](phase2_results.txt) after running `03_model_comparison.py`.

| Model | R² Score | MAE (W/m²) | nRMSE (%) | CV R² |
| :--- | :--- | :--- | :--- | :--- |
| **Random Forest** | **0.9846** 🏆 | **12.75** | **3.96%** | 0.9819 ± 0.0009 |
| **Gradient Boosting** | 0.9841 | 14.90 | 4.02% | 0.9813 ± 0.0008 |
| **XGBoost** | 0.9842 | 13.88 | 4.01% | 0.9815 ± 0.0011 |
| **SVR** | 0.9839 | 16.15 | 4.04% | 0.9818 ± 0.0003 |

*All models significantly outperform the literature baseline of 28% nRMSE. Random Forest remains the best model even with lag features added.*

---

## 📈 Key Visualizations

All 14 generated figures are located in [`plots/`](plots/):

| Figure | Description |
| :--- | :--- |
| **[01_ghi_distribution.png](plots/01_ghi_distribution.png)** | Overall vs Daytime GHI distribution showing nocturnal zeros and daytime peak |
| **[02_hourly_ghi_profile.png](plots/02_hourly_ghi_profile.png)** | 24-hour diurnal profile peaking at ~807 W/m² at solar noon (12:30 PM) |
| **[03_monthly_ghi.png](plots/03_monthly_ghi.png)** | Monthly variation color-coded by Indian seasons (May peak vs Dec trough) |
| **[04_correlation_heatmap.png](plots/04_correlation_heatmap.png)** | Pearson correlation matrix (Temperature +0.59, Humidity -0.29) |
| **[05_seasonal_ghi_profiles.png](plots/05_seasonal_ghi_profiles.png)** | Summer, Winter, Monsoon, and Post-Monsoon comparative diurnal curves |
| **[09_actual_vs_predicted_scatter.png](plots/09_actual_vs_predicted_scatter.png)** | Scatter comparison of measured vs predicted GHI on holdout data |
| **[10_feature_importance.png](plots/10_feature_importance.png)** | Feature importance rankings (Hour cyclical components ~79%, Temp ~12%) |
| **[11_timeseries_7day.png](plots/11_timeseries_7day.png)** | 7-day continuous time-series forecast tracking actual measurements |

---

## 🗂️ Repository Structure

```
├── 01_eda.py                 # Phase 1: Data cleaning, feature engineering & 8 EDA visualizations
├── 02_model.py               # Phase 1: Random Forest regressor, CV, evaluation & 6 diagnostic plots
├── 03_model_comparison.py    # Phase 2: RF vs XGBoost vs GBM vs SVR comparison (8 plots)
├── BA_Combined.xlsx          # Raw dataset: 5 years hourly solar irradiance (Rajasthan)
├── solar_data_cleaned.csv    # Preprocessed dataset with engineered features
├── model_results.txt         # Phase 1: Random Forest evaluation metrics
├── phase2_results.txt        # Phase 2: Full model comparison results table
├── ppt_content.md            # Complete 16-slide PPT content & speaker notes (Phase 1 + 2)
├── dop_project_guide.md      # Team roles, video reference critique, and semester roadmap
├── plots/                    # 22 high-resolution (150 DPI) plots (plots/01–22)
├── .gitignore                # Git ignore rules
└── README.md                 # Project documentation
```

---

## 🚀 How to Run

### 1. Requirements
Ensure Python 3.8+ is installed with the required scientific packages:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn openpyxl
```

### 2. Run Exploratory Data Analysis
```bash
python 01_eda.py
```
*Outputs `solar_data_cleaned.csv` and saves plots `01` through `08` in `plots/`.*

### 3. Train & Evaluate the Baseline Model (Phase 1)
```bash
python 02_model.py
```
*Trains Random Forest Regressor (200 trees, depth 20), performs 5-fold CV, outputs metrics to `model_results.txt`, and generates plots `09` through `14`.*

### 4. Run Model Comparison Study (Phase 2)
```bash
pip install xgboost  # one-time if not installed
python 03_model_comparison.py
```
*Trains RF, Gradient Boosting, XGBoost, and SVR with lag features and time-series CV. Saves results to `phase2_results.txt` and generates plots `15` through `22`.*

---

## 🗓️ Semester Roadmap

```
Phase 1 (Weeks 1-3) [COMPLETED ✓]
├── Static CSV + Random Forest baseline
├── Next-day irradiance prediction (R² = 0.9695, nRMSE = 5.56%)
└── Deliverable: Working baseline & EDA (01_eda.py, 02_model.py)

Phase 2 (Weeks 4-6) [COMPLETED ✓]
├── Lag features + rolling averages (13 → 19 features)
├── Benchmarking: RF vs XGBoost vs Gradient Boosting vs SVR
├── RandomizedSearchCV hyperparameter tuning
├── Time-series cross-validation (TimeSeriesSplit)
└── Deliverable: Model comparison study (03_model_comparison.py)

Phase 3 (Weeks 7-9)
├── Live weather API integration (OpenWeatherMap / NASA POWER API)
├── Automated real-time pipeline for BITS Hyderabad campus (lat: 17.54°N, lon: 78.57°E)
└── Deliverable: Live prediction ingestion pipeline

Phase 4 (Weeks 10-12)
├── Physics-Informed Deep Learning (PINN / BiLSTM / Transformers)
├── Interactive web dashboard (Streamlit / Next.js)
└── Deliverable: End-to-end solar forecasting system
```

---

## 📚 Reference Research Papers

| Year | Title | Authors / Journal | Key Takeaways / Relevance |
| :--- | :--- | :--- | :--- |
| **2023** | *Forecasting Renewable Energy Generation with Machine Learning* | MDPI | Review of AI models for solar & wind forecasting |
| **2024** | *Present and Future of AI in Renewable Energy Domain* | arXiv | Highlights BiLSTM and AB-Net performance |
| **2024** | *Physics-Informed Neural Networks for Solar Wind Prediction* | ResearchGate | Methods for incorporating physical grid constraints into loss functions |
| **Reference Lecture** | *Regression Trees and Solar Radiation Forecasting: Boosting, Bagging, and Ensemble Learning* | Research Lecture | Established 28% nRMSE benchmark for tree ensembles on solar irradiance |
