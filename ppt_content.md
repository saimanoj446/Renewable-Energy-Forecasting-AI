# Forecasting of Renewable Resources Using AI
### PPT Content — Slide-by-Slide Guide
**DOP | BITS Pilani, Hyderabad | EEE Dept | Prof. Arup Ratan**

---

## Slide 1: Title Slide

**Title:** Forecasting of Renewable Resources Using AI

**Subtitle:** Phase 2 — Model Comparison Study: RF vs XGBoost vs Gradient Boosting vs SVR

**Team:** [Your 3 names]

**Course:** Design-Oriented Project (DOP) | Semester: Aug–Dec 2026

**Dept:** Electrical & Electronics Engineering, BITS Pilani, Hyderabad Campus

**Guide:** Prof. Arup Ratan

---

## Slide 2: Problem Statement & Motivation

**Why Solar Forecasting Matters:**
- India's National Solar Mission targets **280 GW** solar capacity by 2030
- Solar energy is intermittent — output depends on weather, time, season
- Accurate forecasting is critical for **grid integration** and **energy planning**
- Poor forecasts lead to grid instability, wasted energy, and financial losses

**Our Goal:**
> Build an AI-based system that predicts **Global Horizontal Irradiance (GHI)** — the total solar radiation received on a horizontal surface — using historical weather data.

**Key Terms:**
| Term | Meaning |
|------|---------|
| GHI | Global Horizontal Irradiance (total solar radiation on flat surface) |
| DNI | Direct Normal Irradiance (direct beam from sun) |
| DHI | Diffuse Horizontal Irradiance (scattered by atmosphere) |
| Relationship | **GHI = DNI × cos(zenith angle) + DHI** |

---

## Slide 3: Literature Review

**Key References:**
1. **"Regression Trees and Solar Radiation Forecasting"** (YouTube lecture, Regression Trees Boosting Bagging study)
   - Compared Random Forest, Bagged Trees, and Boosted Trees
   - RF achieved **28% nRMSE** for 1-hour ahead forecasting
   - Validated RF as a strong baseline for solar prediction

2. **Ahmad et al. (2020)** — "Tree-based ensemble methods for predicting PV power generation"
   - Random Forest outperformed single decision trees by 15-20%
   - Feature importance: Hour of day and temperature are dominant predictors

3. **ML Lifecycle** (Discussed in previous meeting)
   - Data Collection → Preprocessing → Feature Engineering → Model Training → Evaluation → Deployment

> **Note:** Our approach aligns with published research — RF is a proven, strong baseline for solar forecasting before exploring deep learning methods.

---

## Slide 4: System Architecture

**Current (Phase 1) — Static Batch Pipeline:**
```
Excel/CSV Data → Preprocessing → Feature Engineering → Random Forest → Predicted GHI
     ↓                                                        ↓
 Historical         Temperature, Humidity,              Next-day solar
 weather data        Hour, Month, etc.                  irradiance
```

**Future Vision (Phase 4) — Live Pipeline:**
```
Live Weather API → Real-time Pipeline → RF/XGBoost/LSTM → Dashboard
(OpenWeatherMap)    (Automated hourly)   (Model ensemble)   (Web UI)
     ↓                                                        ↓
 Current weather     Auto feature eng.                  Hourly/minutely
 conditions          + data validation                  predictions
```

---

## Slide 5: Dataset Description

**Source:** NASA POWER Database — Rajasthan, India (Solar-rich region)

| Property | Value |
|----------|-------|
| Time Range | 2010–2014 (5 years) |
| Granularity | Hourly (8,760 hours/year) |
| Total Samples | **43,800 rows** |
| Missing Values | **0** (completely clean) |
| Location | Rajasthan, India |

**14 Columns:**
| Category | Features |
|----------|----------|
| Time | Year, Month, Day, Hour, Minute |
| Solar Irradiance | **GHI** (target), DNI, DHI |
| Weather | Temperature, Dew Point, Pressure, Relative Humidity |
| Wind | Wind Direction, Wind Speed |

**Key Statistics:**

| Feature | Mean | Min | Max |
|---------|------|-----|-----|
| GHI (W/m²) | 236.8 | 0 | 991 |
| Temperature (°C) | 28.1 | 4.8 | 51.9 |
| Humidity (%) | 40.7 | 2.8 | 100 |
| Pressure (hPa) | 990.0 | 974.3 | 1006.6 |

---

## Slide 6: EDA — Solar Irradiance Patterns

### Hourly Solar Profile
![Average solar irradiance peaks at 807 W/m² at 12:30 PM, following a smooth bell curve from sunrise (~7AM) to sunset (~18PM)](plots/02_hourly_ghi_profile.png)

**Key Insight:** Peak irradiance of **807 W/m²** at solar noon (12:30). Sunrise ~7AM, sunset ~18PM in Rajasthan.

### Monthly Variation
![Monthly average GHI showing May as the highest (300 W/m²) and December as the lowest (174 W/m²), color-coded by Indian seasons](plots/03_monthly_ghi.png)

**Key Insight:** **May** has highest average GHI (300 W/m²), **December** lowest (174 W/m²). Summer months receive ~72% more solar energy than winter.

---

## Slide 7: EDA — Seasonal & Correlation Analysis

### Seasonal Profiles
![Four seasonal curves showing Summer having the highest peak irradiance (~930 W/m²), followed by Monsoon, Post-Monsoon, and Winter](plots/05_seasonal_ghi_profiles.png)

**Key Insight:** Summer peak is **~930 W/m²** vs Winter peak of **~720 W/m²**. Monsoon shows longer daylight hours but cloud-reduced peak.

### Feature Correlations
![Correlation heatmap showing GHI is highly correlated with DNI (0.94), DHI (0.92), Temperature (0.59), and negatively with Humidity (-0.29)](plots/04_correlation_heatmap.png)

**Key Insights:**
- GHI strongly correlates with **DNI (0.94)** and **DHI (0.92)** — expected, as GHI = DNI·cos(θ) + DHI
- **Temperature (0.59)** positively correlates — hotter days = more solar radiation
- **Relative Humidity (-0.29)** negatively correlates — clouds/moisture block sunlight
- **Pressure vs Dew Point (-0.78)** — strong inverse relationship (meteorological link)

---

## Slide 8: Data Preprocessing & Feature Engineering

**Steps Performed:**
1. **No missing value imputation needed** — dataset is 100% complete
2. **Feature Engineering:**
   - `DayOfYear` (1–365) — captures seasonal patterns
   - `Hour_sin`, `Hour_cos` — cyclical encoding of hour (so hour 23 is close to hour 0)
   - `Month_sin`, `Month_cos` — cyclical encoding of month
3. **Time-based Train/Test Split:**
   - **Training:** 2010–2013 (35,040 samples)
   - **Testing:** 2014 (8,760 samples)
   - *Why time-based?* Prevents data leakage — model never sees future data

**Why Cyclical Encoding?**
```
Regular:  Hour 23 → 0  (appears as big jump)
Cyclical: sin(2π×23/24) ≈ sin(2π×0/24)  (continuous, smooth transition)
```
This helps Random Forest understand that 11 PM and midnight are close in time.

---

## Slide 9: Model — Random Forest Approach

**What is Random Forest?**
- An **ensemble** of 200 decision trees
- Each tree trains on a random subset of data (**bagging**)
- Each split considers a random subset of features
- Final prediction = **average of all 200 trees**

**Why Random Forest for Solar Forecasting?**
- Handles **non-linear relationships** (temperature ↔ irradiance is non-linear)
- Built-in **feature importance** — tells us which weather variables matter most
- **Robust to outliers** and doesn't require feature scaling
- Proven in literature (28% nRMSE, our reference video)

**Hyperparameters:**
| Parameter | Value | Reason |
|-----------|-------|--------|
| n_estimators | 200 | More trees = more stable predictions |
| max_depth | 20 | Prevents overfitting |
| min_samples_split | 10 | Avoids learning noise |
| min_samples_leaf | 5 | Ensures meaningful leaves |
| max_features | sqrt | Standard for regression |

---

## Slide 10: Results — Model Performance

### Metrics Summary
![Performance summary card showing R²=0.9830 on training, R²=0.9695 on test, with MAE of 21 W/m² and RMSE of 55.14 W/m² on test set](plots/13_metrics_summary.png)

### Actual vs Predicted
![Scatter plots showing tight clustering around the perfect prediction line for both training and test sets](plots/09_actual_vs_predicted_scatter.png)

**Talking Points:**
- **R² = 0.9695** — model explains **96.95%** of variance in solar irradiance
- **MAE = 21 W/m²** — average error is only 21 watts per square meter
- **nRMSE = 5.56%** — significantly better than the literature baseline of 28%
- **Train-test gap is small** (0.9830 vs 0.9695) — model is **not overfitting**
- **5-fold CV R² = 0.9652 ± 0.0078** — model is consistent and stable

---

## Slide 11: Results — Feature Importance & Forecast Demo

### What Drives Solar Irradiance?
![Feature importance chart showing Hour_cos (0.519) and Hour (0.206) dominate, followed by Temperature (0.119)](plots/10_feature_importance.png)

**Key Finding:** **Time of day (Hour)** contributes **79%** of prediction power, followed by **Temperature (12%)**. This makes physical sense — solar position determines irradiance.

### 7-Day Forecast Sample
![Time series showing predicted vs actual GHI for March 10-16, 2014 with very close alignment across all 7 days](plots/11_timeseries_7day.png)

**Key Finding:** The model accurately tracks the daily solar cycle for an entire week. Predictions closely follow actual measurements, including variations in peak irradiance across days.

---

## Slide 12: Phase 2 — Feature Engineering Advances

**What's New in Phase 2 (Beyond Phase 1)?**

| Feature Type | Features Added | Why It Helps |
|---|---|---|
| **Lag-1 GHI** | GHI 1 hour ago | Solar radiation is highly autocorrelated — yesterday's hour predicts today's |
| **Lag-2, Lag-3 GHI** | GHI 2h and 3h ago | Captures short-term weather trends |
| **3-hour rolling avg** | Mean of last 3h GHI | Smooths noise, encodes recent trend direction |
| **6-hour rolling avg** | Mean of last 6h GHI | Captures half-day weather patterns |
| **Temp Lag-1** | Temperature 1h ago | Thermal inertia — temperature changes gradually |

**Result:** Feature count grew from **13 → 19**. Lag features made GHI_lag1 the **top predictor** in all tree models (importance > 0.40).

**Time-Series Cross-Validation (TimeSeriesSplit)**
- Unlike random k-fold, always trains on **past data only** and tests on **future data**
- 5 chronological folds on the 2010–2013 training set
- Ensures CV scores truly reflect real-world deployment performance

---

## Slide 13: Phase 2 — Model Comparison Results

**Models Trained & Tuned:**
| Model | Tuning Method | Key Hyperparameters |
|---|---|---|
| Random Forest | Fixed (Phase 1) | 200 trees, depth=20 |
| Gradient Boosting | RandomizedSearchCV (15 trials) | LR, depth, n_estimators, subsample |
| XGBoost | RandomizedSearchCV (15 trials) | LR, depth, colsample, reg_alpha |
| SVR | Fixed (RBF kernel) | C=500, gamma=scale |

**Comparison Table (Test Set — 2014):**

| Model | R² Score | MAE (W/m²) | nRMSE (%) | CV R² |
|---|---|---|---|---|
| **Random Forest** 🏆 | **0.9846** | **12.75** | **3.96%** | 0.9819 ± 0.0009 |
| **Gradient Boosting** | 0.9841 | 14.90 | 4.02% | 0.9813 ± 0.0008 |
| **XGBoost** | 0.9842 | 13.88 | 4.01% | 0.9815 ± 0.0011 |
| **SVR** | 0.9839 | 16.15 | 4.04% | 0.9818 ± 0.0003 |

> **💡 Key Finding:** All 4 models achieve R² > 0.98 — confirming solar irradiance is highly predictable with good features. **Random Forest remains the best model** even after adding lag features. Phase 1 RF (nRMSE=5.56%) improved to **3.96%** in Phase 2 thanks to lag features alone.

**Key Plots to show:**
- `plots/15_model_comparison_metrics.png` — Bar chart: R², MAE, nRMSE
- `plots/16_cv_comparison.png` — Boxplot: 5-fold CV stability
- `plots/17_7day_forecast_comparison.png` — All 4 models on same 7-day window
- `plots/21_phase2_summary_card.png` — Summary card with best model

---

## Slide 14: Semester Roadmap (Updated)

```
Phase 1 (Week 1-3) — COMPLETED ✓
├── Static CSV + Random Forest baseline
├── R² = 0.9695, nRMSE = 5.56%
└── Deliverable: Working baseline model

Phase 2 (Week 4-6) — COMPLETED ✓
├── Lag features + rolling averages (19 features)
├── RF vs XGBoost vs Gradient Boosting vs SVR
├── RandomizedSearchCV hyperparameter tuning
├── Time-series cross-validation (TimeSeriesSplit)
└── Deliverable: Model comparison report

Phase 3 (Week 7-9)
├── Live weather API integration (OpenWeatherMap / NASA POWER API)
├── Real-time data pipeline for BITS Hyderabad campus
├── Automated hourly predictions for Hyderabad location
└── Deliverable: Live prediction pipeline

Phase 4 (Week 10-12)
├── BiLSTM / Transformer / PINN deep learning models
├── Interactive web dashboard (Streamlit / Next.js)
├── Hyderabad campus-specific predictions
├── Final report + live demo
└── Deliverable: Complete end-to-end system with UI
```

---

## Slide 15: Next Steps (Phase 3 Preview)

**Phase 3 Goals (Weeks 7–9):**
1. **NASA POWER API** — fetch Hyderabad-specific live weather data automatically
2. **OpenWeatherMap API** — real-time temperature, humidity, cloud cover
3. **Automated Pipeline** — scheduled script that runs predictions every hour
4. **BITS Hyderabad Model** — retrain best Phase 2 model on Hyderabad coordinates

**Questions for Sir:**
- Confirm which API to prioritize: NASA POWER (historical accuracy) or OpenWeatherMap (real-time)?
- Should the Phase 3 pipeline predict for BITS Hyderabad campus specifically (lat: 17.54°N, lon: 78.57°E)?
- Any performance threshold to meet before moving to Phase 4 deep learning?

---

## Slide 16: Thank You & Q&A

**Phase 2 Summary:**
- Benchmarked **4 models** — RF, XGBoost, Gradient Boosting, SVR
- Added **6 lag/rolling features** — GHI autocorrelation is the strongest predictor
- Used **time-series cross-validation** — honest evaluation on future data
- **Hyperparameter tuning** via RandomizedSearchCV (15 trials each)
- Results saved in `phase2_results.txt` and plots `15–22` in `plots/`

**GitHub / Code:** [Link to your repo]

**References:**
1. YouTube: "Regression Trees and Solar Radiation Forecasting" (Boosting, Bagging, Ensemble)
2. NASA POWER Database (data source)
3. scikit-learn & XGBoost documentation
4. Chen & Guestrin (2016) — XGBoost: A Scalable Tree Boosting System
