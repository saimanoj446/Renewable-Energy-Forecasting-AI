# Forecasting of Renewable Resources Using AI: A Machine Learning Approach for Short-Term Solar Irradiance Prediction

**Design-Oriented Project (DOP) — Midsemester Report**

**Authors:** [Name 1 (ID)], [Name 2 (ID)], [Name 3 (ID)]

**Department:** Electrical & Electronics Engineering, BITS Pilani, Hyderabad Campus

**Guide:** Prof. Arup Ratan

**Date:** September 2026

---

## Abstract

> This report presents a machine learning framework for short-term Global Horizontal Irradiance (GHI) forecasting using five years of hourly solar radiation data from Rajasthan, India. We develop a two-phase methodology: Phase 1 establishes a Random Forest baseline achieving R² = 0.9695 and nRMSE = 5.56% on a holdout test year; Phase 2 introduces temporal lag features and benchmarks four regression models — Random Forest, Gradient Boosting, XGBoost, and Support Vector Regression (SVR). With 19 engineered features including lag-based autocorrelations, all models achieve R² > 0.98, with Random Forest delivering the best performance (R² = 0.9846, nRMSE = 3.96%). These results significantly outperform the literature baseline of 28% nRMSE reported for tree-based ensemble methods. The report also outlines the planned phases for live API integration and deep learning model deployment.

> **Keywords:** Solar irradiance forecasting, GHI prediction, Random Forest, XGBoost, ensemble learning, feature engineering, time-series cross-validation

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Literature Review](#2-literature-review)
3. [Dataset Description](#3-dataset-description)
4. [Methodology](#4-methodology)
5. [Exploratory Data Analysis](#5-exploratory-data-analysis)
6. [Results & Discussion](#6-results--discussion)
7. [Future Work](#7-future-work)
8. [Conclusion](#8-conclusion)
9. [References](#9-references)

---

## 1. Introduction

### 1.1 Background & Motivation

India's National Solar Mission targets 280 GW of installed solar capacity by 2030, making accurate solar energy forecasting a critical component of the country's renewable energy strategy. Solar power generation is inherently intermittent, as output depends on meteorological conditions including cloud cover, temperature, humidity, and the sun's position. This intermittency poses challenges for grid operators responsible for unit commitment, dispatch scheduling, and maintaining frequency stability.

Global Horizontal Irradiance (GHI) — the total solar radiation received per unit area on a horizontal surface — is the primary metric used to quantify available solar energy at a location. GHI is composed of two components:

**GHI = DNI × cos(θ_z) + DHI**

where DNI is the Direct Normal Irradiance, DHI is the Diffuse Horizontal Irradiance, and θ_z is the solar zenith angle.

Accurate short-term GHI forecasting (1–24 hours ahead) directly impacts grid scheduling efficiency, curtailment reduction, and the economic viability of solar plants. This project addresses this challenge through a data-driven, multi-phase machine learning framework.

### 1.2 Objectives

1. Develop a robust baseline model for hourly GHI prediction using historical meteorological data
2. Compare multiple ensemble regression algorithms for solar irradiance forecasting
3. Investigate the impact of temporal lag features on forecasting accuracy
4. Establish a semester-long roadmap progressing from static CSV-based models to real-time, physics-informed deep learning systems

### 1.3 Scope of This Report

This midsemester report covers the work completed in **Phase 1** (Random Forest baseline) and **Phase 2** (multi-model comparison with lag features). Phases 3 (live API integration) and 4 (deep learning / dashboard) are planned for the remaining semester.

---

## 2. Literature Review

Machine learning approaches for solar irradiance forecasting have been extensively studied in recent literature. The key references that inform our approach are summarized below.

### 2.1 Ensemble Methods for Solar Forecasting

Tree-based ensemble methods such as Random Forest, Gradient Boosting, and XGBoost have emerged as strong performers for solar irradiance prediction tasks. Ahmad et al. [1] demonstrated that Random Forest outperforms single decision trees by 15–20% for PV power generation prediction, while Chen & Guestrin [2] showed that XGBoost provides superior accuracy and computational efficiency through its regularized gradient boosting framework.

A comparative study by Javed et al. [3] on solar irradiance prediction using linear regression, decision trees, and SVR found that non-linear machine learning approaches are significantly better suited for capturing the complex relationships between meteorological variables and solar radiation.

### 2.2 Feature Engineering & Temporal Dependencies

Recent research emphasizes that solar radiation forecasting is fundamentally a time-series task, and incorporating lag features (past GHI observations) significantly improves model performance [4]. The autocorrelation structure of GHI — where the irradiance at a given hour is strongly correlated with the preceding hours — can be exploited through rolling averages and lagged variables.

Cyclical encoding of temporal features (hour, month) using sinusoidal transformations has been shown to improve model performance by preserving the periodic nature of these variables [5].

### 2.3 Research Baseline

A referenced study on regression trees and solar radiation forecasting [6] reported approximately 28% nRMSE for 1-hour ahead solar irradiance prediction using Random Forest. This serves as our benchmark for evaluating model improvements.

### 2.4 Physics-Informed Approaches

For our future phases, we draw inspiration from recent work on Physics-Informed Neural Networks (PINNs) for solar forecasting [7], which embed physical constraints (such as atmospheric and geometric laws) into neural network loss functions to ensure physically plausible predictions.

---

## 3. Dataset Description

### 3.1 Data Source

We use hourly solar irradiance data from the **NASA POWER Database** for Rajasthan, India — one of India's highest solar potential regions. The dataset spans 5 years (2010–2014) with 43,800 hourly observations and zero missing values.

### 3.2 Features

| Category | Features | Description |
|---|---|---|
| **Time** | Year, Month, Day, Hour, Minute | Temporal identifiers |
| **Solar Irradiance** | GHI (target), DNI, DHI | Radiation components (W/m²) |
| **Weather** | Temperature, Dew Point, Pressure, Relative Humidity | Meteorological conditions |
| **Wind** | Wind Direction, Wind Speed | Wind characteristics |

### 3.3 Summary Statistics

| Feature | Mean | Std Dev | Min | Max |
|---|---|---|---|---|
| GHI (W/m²) | 236.8 | 304.2 | 0 | 991 |
| Temperature (°C) | 28.1 | 8.4 | 4.8 | 51.9 |
| Relative Humidity (%) | 40.7 | 25.6 | 2.8 | 100.0 |
| Pressure (hPa) | 990.0 | 6.1 | 974.3 | 1006.6 |
| Wind Speed (m/s) | 3.2 | 2.1 | 0.0 | 15.4 |

**Note:** GHI has a mean of 236.8 W/m² across all hours (including nighttime zeros). Daytime-only GHI (when GHI > 0) averages approximately 470 W/m².

---

## 4. Methodology

### 4.1 Data Preprocessing & Feature Engineering

#### 4.1.1 Phase 1 Features (13 features)

Starting from the raw 14-column dataset, we engineered the following features:

- **DayOfYear** (1–365): Captures seasonal position within the year
- **Hour_sin, Hour_cos**: Cyclical encoding of hour using sinusoidal transformation:

  ```
  Hour_sin = sin(2π × Hour / 24)
  Hour_cos = cos(2π × Hour / 24)
  ```

  This encoding ensures temporal continuity — hour 23 and hour 0 are treated as adjacent, unlike raw integer encoding.

- **Month_sin, Month_cos**: Cyclical encoding of month using the same approach

**Complete Phase 1 feature set (13 features):**
Hour, Month, DayOfYear, Temperature, Dew Point, Pressure, Relative Humidity, Wind Speed, Wind Direction, Hour_sin, Hour_cos, Month_sin, Month_cos

#### 4.1.2 Phase 2 Features (19 features = 13 original + 6 new)

In Phase 2, we augmented the feature set with temporal lag and rolling statistics:

| New Feature | Definition | Rationale |
|---|---|---|
| **GHI_lag1** | GHI value 1 hour prior | Solar radiation is highly autocorrelated |
| **GHI_lag2** | GHI value 2 hours prior | Captures short-term weather trend |
| **GHI_lag3** | GHI value 3 hours prior | Extended temporal context |
| **GHI_roll3** | 3-hour rolling mean of GHI | Smooths noise, encodes recent trend |
| **GHI_roll6** | 6-hour rolling mean of GHI | Half-day weather pattern |
| **Temp_lag1** | Temperature 1 hour prior | Thermal inertia effect |

#### 4.1.3 Train/Test Split

We use a **chronological time-based split** (not random splitting):
- **Training set:** 2010–2013 (35,040 samples, ~80%)
- **Test set:** 2014 (8,760 samples, ~20%)

This prevents data leakage — the model never sees future data during training, which mirrors real-world deployment conditions.

### 4.2 Phase 1 — Random Forest Baseline

#### 4.2.1 Model Architecture

Random Forest is an ensemble of decision trees that uses **bagging** (bootstrap aggregation) to reduce variance. Each tree is trained on a random bootstrap sample of the training data, and each split considers a random subset of features. The final prediction is the average of all tree predictions.

#### 4.2.2 Hyperparameters

| Parameter | Value | Justification |
|---|---|---|
| `n_estimators` | 200 | Sufficient trees for stable ensemble averaging |
| `max_depth` | 20 | Controls tree complexity; prevents overfitting |
| `min_samples_split` | 10 | Avoids splitting on noisy, small groups |
| `min_samples_leaf` | 5 | Ensures each leaf has meaningful sample support |
| `max_features` | sqrt(n) | Standard for regression; promotes tree diversity |

#### 4.2.3 Evaluation Metrics

We evaluate all models using four standard regression metrics:

- **R² (Coefficient of Determination):** Proportion of variance explained (1.0 = perfect)
- **MAE (Mean Absolute Error):** Average absolute prediction error in W/m²
- **RMSE (Root Mean Squared Error):** Penalizes large errors more heavily than MAE
- **nRMSE (Normalized RMSE):** RMSE normalized by the range of the target variable (%), enabling comparison across datasets

Additionally, we report **5-fold cross-validation R²** using `TimeSeriesSplit` to ensure temporal integrity.

### 4.3 Phase 2 — Multi-Model Comparison

#### 4.3.1 Models Evaluated

| Model | Tuning Strategy | Key Hyperparameters |
|---|---|---|
| **Random Forest** | Fixed (from Phase 1) | 200 trees, depth=20 |
| **Gradient Boosting** | RandomizedSearchCV (15 iterations) | Learning rate, depth, n_estimators, subsample |
| **XGBoost** | RandomizedSearchCV (15 iterations) | Learning rate, depth, colsample_bytree, reg_alpha |
| **SVR (RBF kernel)** | Fixed | C=500, gamma=scale |

#### 4.3.2 Cross-Validation Strategy

We use **TimeSeriesSplit** with 5 folds on the training set (2010–2013). Unlike standard k-fold CV, TimeSeriesSplit ensures that training always precedes testing chronologically:

```
Fold 1: Train [2010]       → Test [2011 Q1]
Fold 2: Train [2010-2011]  → Test [2011 Q3]
Fold 3: Train [2010-2011]  → Test [2012 Q1]
Fold 4: Train [2010-2012]  → Test [2012 Q3]
Fold 5: Train [2010-2012]  → Test [2013 Q1]
```

---

## 5. Exploratory Data Analysis

### 5.1 GHI Distribution

The GHI distribution is heavily zero-inflated due to nighttime hours (approximately 50% of all observations have GHI = 0). Daytime GHI follows a roughly uniform distribution between 50–950 W/m², with a slight peak around 600–800 W/m².

*[Insert Figure: plots/01_ghi_distribution.png — caption: "Distribution of GHI across all hours (left) and daytime-only hours (right)"]*

### 5.2 Diurnal Profile

The average hourly GHI follows a smooth bell curve, rising from approximately 7:00 AM, peaking at **807 W/m² at 12:30 PM** (solar noon), and declining to zero by 6:00 PM. This diurnal pattern is the single strongest signal in the data.

*[Insert Figure: plots/02_hourly_ghi_profile.png — caption: "Average hourly GHI profile (2010–2014)"]*

### 5.3 Seasonal Variation

Monthly average GHI ranges from **174 W/m² in December** (winter) to **300 W/m² in May** (pre-monsoon summer). Summer months receive approximately 72% more solar energy than winter months. The Indian monsoon season (June–September) shows reduced peak irradiance due to cloud cover despite longer daylight hours.

*[Insert Figure: plots/03_monthly_ghi.png — caption: "Monthly average GHI color-coded by Indian seasons"]*

### 5.4 Feature Correlations

Key correlations with GHI:
- **DNI (+0.94)** and **DHI (+0.92):** Expected, as GHI = DNI·cos(θ_z) + DHI
- **Temperature (+0.59):** Higher temperatures correlate with clearer skies and more irradiance
- **Relative Humidity (−0.29):** Moisture and clouds reduce incoming solar radiation
- **Pressure vs. Dew Point (−0.78):** Strong inverse meteorological relationship

*[Insert Figure: plots/04_correlation_heatmap.png — caption: "Pearson correlation matrix of meteorological and solar features"]*

### 5.5 Seasonal Irradiance Profiles

Summer peak irradiance reaches ~930 W/m² compared to Winter's ~720 W/m². The monsoon season shows broader but lower diurnal curves due to persistent cloud cover.

*[Insert Figure: plots/05_seasonal_ghi_profiles.png — caption: "Seasonal diurnal GHI profiles"]*

---

## 6. Results & Discussion

### 6.1 Phase 1 — Random Forest Baseline Results

Trained on 2010–2013 (35,040 samples), tested on 2014 (8,760 samples):

| Metric | Training Set (2010–2013) | Test Set (2014) |
|---|---|---|
| **R² Score** | 0.9830 | **0.9695** |
| **MAE** | 15.03 W/m² | **21.00 W/m²** |
| **RMSE** | 40.89 W/m² | **55.14 W/m²** |
| **nRMSE** | 4.13% | **5.56%** |
| **5-Fold CV R²** | — | **0.9652 ± 0.0078** |

**Key observations:**
- The model explains **96.95%** of variance in the test year, with an average prediction error of only 21 W/m²
- The small train-test gap (R² 0.9830 vs 0.9695) indicates the model is **not overfitting**
- The nRMSE of **5.56% significantly outperforms the literature baseline of 28%** reported for RF-based solar forecasting [6]
- Cross-validation R² of 0.9652 ± 0.0078 confirms consistent performance across different time periods

*[Insert Figure: plots/09_actual_vs_predicted_scatter.png — caption: "Actual vs Predicted GHI for training (left) and test (right) sets"]*

#### 6.1.1 Feature Importance (Phase 1)

| Rank | Feature | Importance |
|---|---|---|
| 1 | Hour_cos | 0.5190 |
| 2 | Hour | 0.2061 |
| 3 | Temperature | 0.1189 |
| 4 | Hour_sin | 0.0642 |
| 5 | Relative Humidity | 0.0380 |
| 6–13 | Others (DayOfYear, Wind, Pressure, etc.) | < 0.01 each |

**Observation:** Time-of-day features (Hour_cos + Hour + Hour_sin) collectively contribute **~79%** of the model's predictive power, followed by Temperature at **~12%**. This aligns with the physical understanding that solar position (determined by hour) is the dominant driver of surface irradiance.

*[Insert Figure: plots/10_feature_importance.png — caption: "Random Forest feature importance (Gini impurity)"]*

### 6.2 Phase 2 — Multi-Model Comparison Results

With 19 features (13 original + 6 lag/rolling features), all four models were evaluated on the 2014 test set:

| Model | R² Score | MAE (W/m²) | RMSE (W/m²) | nRMSE (%) | CV R² |
|---|---|---|---|---|---|
| **Random Forest** 🏆 | **0.9846** | **12.75** | **39.21** | **3.96%** | 0.9819 ± 0.0009 |
| XGBoost | 0.9842 | 13.88 | 39.70 | 4.01% | 0.9815 ± 0.0011 |
| Gradient Boosting | 0.9841 | 14.90 | 39.88 | 4.02% | 0.9813 ± 0.0008 |
| SVR | 0.9839 | 16.15 | 40.04 | 4.04% | 0.9818 ± 0.0003 |

**Key findings:**

1. **All four models achieve R² > 0.98**, confirming that solar irradiance is highly predictable given good feature engineering
2. **Random Forest is the best model** across all metrics — lowest MAE (12.75 W/m²), lowest nRMSE (3.96%), and highest R² (0.9846)
3. **Lag features provide a massive improvement:** Phase 1 RF (nRMSE = 5.56%) improved to **3.96%** in Phase 2 — a **28.8% reduction** in normalized error, driven entirely by the addition of 6 lag/rolling features
4. **The performance gap between models is small** (< 0.1% in R²), suggesting that feature engineering matters more than algorithm selection for this problem
5. **SVR shows the most stable CV** (std = 0.0003), but slightly lower overall accuracy

*[Insert Figure: plots/15_model_comparison_metrics.png — caption: "Comparative bar chart of R², MAE, and nRMSE across four models"]*

*[Insert Figure: plots/17_7day_forecast_comparison.png — caption: "7-day GHI forecast overlay for all four models"]*

### 6.3 Impact of Lag Features — Phase 1 vs Phase 2

| Metric | Phase 1 (13 features) | Phase 2 (19 features) | Improvement |
|---|---|---|---|
| R² | 0.9695 | 0.9846 | +1.56% |
| MAE | 21.00 W/m² | 12.75 W/m² | −39.3% |
| nRMSE | 5.56% | 3.96% | −28.8% |

The addition of GHI lag features (GHI_lag1, GHI_lag2, GHI_lag3) and rolling averages (GHI_roll3, GHI_roll6) transformed GHI autocorrelation into the **dominant predictor**. In Phase 2, GHI_lag1 became the single most important feature (importance > 0.40) across all tree-based models, displacing Hour_cos from its Phase 1 position.

This confirms the well-established finding in time-series forecasting that **recent historical values of the target variable carry more predictive power than exogenous features** for short-term horizons.

---

## 7. Future Work

### 7.1 Phase 3 — Live API Integration (Weeks 7–9)

- Integrate **NASA POWER API** and **OpenWeatherMap API** for real-time weather data ingestion
- Build an automated pipeline for BITS Pilani Hyderabad campus (17.54°N, 78.57°E)
- Retrain the best Phase 2 model on Hyderabad-specific coordinates

### 7.2 Phase 4 — Deep Learning & Deployment (Weeks 10–12)

- Implement **BiLSTM** (Bidirectional Long Short-Term Memory) for temporal sequence learning
- Explore **Temporal Transformers** for attention-based solar forecasting
- Investigate **Physics-Informed Neural Networks (PINNs)** that embed atmospheric constraints into the loss function
- Deploy an interactive **web dashboard** (Streamlit or Next.js) for real-time GHI predictions

---

## 8. Conclusion

This midsemester report demonstrates a systematic, two-phase machine learning framework for short-term solar irradiance forecasting. Key contributions include:

1. **Phase 1** established a strong Random Forest baseline (R² = 0.9695, nRMSE = 5.56%) that already outperforms the literature benchmark of 28% nRMSE for tree-based methods
2. **Phase 2** showed that temporal feature engineering (lag and rolling features) provides a 28.8% reduction in nRMSE — a larger improvement than any algorithm change
3. A comprehensive comparison of four models (RF, XGBoost, Gradient Boosting, SVR) confirms that **Random Forest delivers the best overall accuracy** at 3.96% nRMSE, though all four models achieve competitive R² > 0.98
4. The project establishes a clear roadmap from static batch prediction to real-time, physics-informed forecasting for campus-level deployment

These results validate the viability of ensemble machine learning methods for operational solar forecasting in Indian climatic conditions.

---

## 9. References

[1] T. Ahmad, H. Chen, and Y. Shah, "Tree-based ensemble methods for predicting PV power generation and their comparison with deep learning approaches," *IEEE Access*, vol. 8, pp. 73–87, 2020.

[2] T. Chen and C. Guestrin, "XGBoost: A Scalable Tree Boosting System," in *Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining*, pp. 785–794, 2016.

[3] A. Javed, H. Larijani, and A. Wixted, "Predicting solar irradiance using machine learning techniques," in *Proc. IEEE Int. Symp. on Wireless Systems within the Conferences on Intelligent Data Acquisition*, pp. 1–6, 2020.

[4] R. H. Inman, H. T. C. Pedro, and C. F. M. Coimbra, "Solar forecasting methods for renewable energy integration," *Progress in Energy and Combustion Science*, vol. 39, no. 6, pp. 535–576, 2013.

[5] P. Kumari and D. Toshniwal, "Deep learning models for solar irradiance forecasting: A comprehensive review," *Journal of Cleaner Production*, vol. 318, 2021.

[6] "Regression Trees and Solar Radiation Forecasting: Boosting, Bagging, and Ensemble Learning Cases," Research Lecture on Solar Forecasting Methods, 2022.

[7] M. Raissi, P. Perdikaris, and G. E. Karniadakis, "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations," *Journal of Computational Physics*, vol. 378, pp. 686–707, 2019.

[8] NASA POWER Data Access Viewer, "Prediction of Worldwide Energy Resources," NASA Langley Research Center. [Online]. Available: https://power.larc.nasa.gov/

---

## Appendix A: Code Repository

The complete source code for this project is available at:
**GitHub:** https://github.com/saimanoj446/Renewable-Energy-Forecasting-AI

| File | Description |
|---|---|
| `01_eda.py` | Data cleaning, feature engineering & 8 EDA visualizations |
| `02_model.py` | Random Forest regressor, cross-validation, evaluation & 6 diagnostic plots |
| `03_model_comparison.py` | Phase 2: RF vs XGBoost vs GBM vs SVR (8 comparison plots) |

---

## Appendix B: List of Figures

| Figure No. | Description | File |
|---|---|---|
| Fig. 1 | GHI Distribution (Overall vs Daytime) | `plots/01_ghi_distribution.png` |
| Fig. 2 | Average Hourly GHI Profile | `plots/02_hourly_ghi_profile.png` |
| Fig. 3 | Monthly Average GHI (Season-coded) | `plots/03_monthly_ghi.png` |
| Fig. 4 | Feature Correlation Heatmap | `plots/04_correlation_heatmap.png` |
| Fig. 5 | Seasonal GHI Profiles | `plots/05_seasonal_ghi_profiles.png` |
| Fig. 6 | Actual vs Predicted Scatter | `plots/09_actual_vs_predicted_scatter.png` |
| Fig. 7 | Feature Importance Rankings | `plots/10_feature_importance.png` |
| Fig. 8 | 7-Day Time-Series Forecast | `plots/11_timeseries_7day.png` |
| Fig. 9 | Residual Analysis | `plots/12_residual_analysis.png` |
| Fig. 10 | Phase 2 Model Comparison | `plots/15_model_comparison_metrics.png` |
| Fig. 11 | Phase 2 7-Day Forecast Comparison | `plots/17_7day_forecast_comparison.png` |
| Fig. 12 | Phase 2 Summary Card | `plots/21_phase2_summary_card.png` |
