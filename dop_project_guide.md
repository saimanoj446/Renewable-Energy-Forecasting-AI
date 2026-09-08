# DOP: Forecasting of Renewable Resources Using AI
**BITS Hyderabad | EEE Dept | Under Arup Ratan Sir**

---

## 1. Is the YouTube Video a Good Reference?

**Video:** *"Regression Trees and Solar Radiation Forecasting: the Boosting, Bagging and Ensemble Learning Cases"*

**Verdict: Yes, it's a solid theoretical reference, but NOT a coding tutorial.**

| Pros | Cons |
|------|------|
| Directly compares Random Forest vs Bagged vs Boosted trees for solar forecasting | It's a research presentation, not hands-on code |
| Shows real performance metrics (28% nRMSE for 1-hr horizon with RF) | Won't teach you how to implement in Python |
| Covers data quality checks, clear-sky modeling, stationarizing the time series | Assumes ML background |
| Validates that Random Forest is a strong baseline — exactly your approach | |

> [!TIP]
> **Use this video for:** Justifying your model choice in the PPT ("literature shows RF achieves 28% nRMSE for solar forecasting"). For actual coding, refer to scikit-learn's `RandomForestRegressor` documentation and Kaggle notebooks.

---

## 2. Is Tomorrow's Goal Plausible? — **Absolutely YES** ✅

Predicting next-day solar irradiance from a static CSV using Random Forest is a **classic, well-studied ML problem**. Here's why it's perfect as a baseline:

- **Random Forest on tabular data** is one of the fastest models to get working (< 50 lines of Python)
- **Static CSV** removes all API/infra complexity — pure ML focus
- **Next-day prediction** is a standard forecasting horizon in literature
- **You already discussed the ML lifecycle** last week — this is just executing Step 1

**Expected results you can show tomorrow:**
- Train/test split on historical data
- R², MAE, RMSE metrics
- A plot: actual vs predicted irradiance
- Feature importance chart (which weather variables matter most)

---

## 3. Recommended Datasets

### 🏆 Best Pick for Your Project

| Dataset | Why It's Great | Link |
|---------|---------------|------|
| **Solar Radiation Prediction — Rajasthan, India** | 5 years hourly data (2010–2014), India-specific, designed for ML | [Kaggle Link](https://www.kaggle.com/datasets/aishwaryamathur/solar-radiation-prediction-dataset-rajasthan-in) |
| **Time Series Solar Irradiance for Indian Cities** | Hourly data for Ahmedabad, Bengaluru, Mumbai (NASA POWER source) | [Kaggle Link](https://www.kaggle.com/datasets/meeenaxi/time-series-solar-irradiance-for-indian-cities) |
| **Solar Data from Diverse Regions** | 2024 data, includes GHI/DNI/DHI + power output | [Kaggle Link](https://www.kaggle.com/datasets/prathameshgadekar/solar-data-from-diverse-regions) |

> [!IMPORTANT]
> **Go with the Rajasthan dataset** — it's the most ML-ready, India-specific, and has 5 years of hourly data which is perfect for your semester-long progression from daily → hourly → minutely.

### Typical Features in These Datasets
- **Target:** GHI (Global Horizontal Irradiance) in W/m²
- **Input Features:** Temperature, Humidity, Wind Speed, Pressure, Cloud Cover, Hour, Month, Day of Year

### Alternative Data Source
If you want Hyderabad-specific data later, use **[NASA POWER Data Access Viewer](https://power.larc.nasa.gov/data-access-viewer/)** — free, precise, and lets you download custom date ranges for any lat/long.

---

## 4. Work Delegation (3 People)

### 👤 Person 1: **Data Engineer** — *Data Collection & Preprocessing*
**Timeline: Tonight + ongoing**

- [ ] Download the Rajasthan dataset from Kaggle
- [ ] Exploratory Data Analysis (EDA):
  - Shape, dtypes, missing values, describe()
  - Distribution plots for irradiance
  - Correlation heatmap
- [ ] Data cleaning: handle missing values, outliers
- [ ] Feature engineering: extract `hour`, `month`, `day_of_year` from timestamp
- [ ] Save cleaned CSV
- [ ] Create 2-3 EDA slides for the PPT

### 👤 Person 2: **ML Engineer** — *Model Building & Evaluation*
**Timeline: Tonight + ongoing**

- [ ] Train/test split (80/20, or time-based split)
- [ ] Build `RandomForestRegressor` with scikit-learn
- [ ] Hyperparameter tuning (`n_estimators`, `max_depth`, `min_samples_split`)
- [ ] Evaluate: R², MAE, RMSE, MAPE
- [ ] Generate plots:
  - Actual vs Predicted scatter/line plot
  - Feature importance bar chart
  - Residual distribution
- [ ] Create 3-4 results slides for the PPT

### 👤 Person 3: **Presenter/Integrator** — *PPT, Roadmap & Literature*
**Timeline: Tonight + ongoing**

- [ ] PPT structure and design
- [ ] Literature review slides (cite the YouTube video, 2-3 papers)
- [ ] Project roadmap slide showing semester progression
- [ ] Architecture diagram (current static CSV flow → future live API flow)
- [ ] Integrate outputs from Person 1 & 2 into the PPT
- [ ] Practice the presentation flow

---

## 5. Semester Roadmap (Show This in PPT)

This demonstrates your incremental improvement plan to Arup Ratan sir:

```
Phase 1 (Week 1-3) ← YOU ARE HERE
├── Static CSV + Random Forest
├── Next-day irradiance prediction  
├── Baseline metrics (R², RMSE)
└── Deliverable: Working baseline model

Phase 2 (Week 4-6)
├── Hourly prediction granularity
├── Compare RF vs XGBoost vs Gradient Boosting
├── Time-series cross-validation
└── Deliverable: Model comparison report

Phase 3 (Week 7-9)
├── Live weather API integration (OpenWeatherMap / NASA POWER API)
├── Real-time data pipeline
├── Minutely/hourly predictions
└── Deliverable: Live prediction pipeline

Phase 4 (Week 10-12)
├── LSTM / hybrid model comparison
├── Dashboard / web interface
├── Hyderabad-specific predictions
├── Final report + demo
└── Deliverable: Complete system with UI
```

---

## 6. Quick-Start Code Skeleton

Here's the minimal code Person 2 needs to get started tonight:

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 1. Load data
df = pd.read_csv('solar_data.csv')

# 2. Feature engineering (Person 1 prepares this)
# Typical features: Temperature, Humidity, WindSpeed, Pressure, Hour, Month
features = ['Temperature', 'Humidity', 'WindSpeed', 'Pressure', 'Hour', 'Month']
target = 'GHI'  # Global Horizontal Irradiance

X = df[features]
y = df[target]

# 3. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train Random Forest
rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
rf.fit(X_train, y_train)

# 5. Predict & Evaluate
y_pred = rf.predict(X_test)
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f} W/m²")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f} W/m²")

# 6. Feature Importance Plot
feat_imp = pd.Series(rf.feature_importances_, index=features).sort_values()
feat_imp.plot(kind='barh', title='Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')

# 7. Actual vs Predicted
plt.figure(figsize=(10, 5))
plt.plot(y_test.values[:100], label='Actual', alpha=0.8)
plt.plot(y_pred[:100], label='Predicted', alpha=0.8)
plt.legend()
plt.title('Solar Irradiance: Actual vs Predicted')
plt.ylabel('GHI (W/m²)')
plt.savefig('actual_vs_predicted.png')
```

> [!NOTE]
> Adjust the column names based on the actual dataset you download. The Rajasthan dataset may use different column headers — Person 1 should standardize these during preprocessing.

---

## 7. PPT Structure (Suggested ~12 slides)

| # | Slide | Owner |
|---|-------|-------|
| 1 | Title: "Forecasting of Renewable Resources Using AI" | P3 |
| 2 | Problem Statement & Motivation | P3 |
| 3 | Literature Review (cite YT video + 2 papers) | P3 |
| 4 | Project Architecture (current + future vision) | P3 |
| 5 | Dataset Description | P1 |
| 6 | EDA: Key Visualizations | P1 |
| 7 | Data Preprocessing Steps | P1 |
| 8 | Model: Random Forest Approach | P2 |
| 9 | Results: Metrics (R², RMSE, MAE) | P2 |
| 10 | Results: Actual vs Predicted + Feature Importance | P2 |
| 11 | Semester Roadmap (Phase 1→4) | P3 |
| 12 | Next Steps & Q&A | All |
