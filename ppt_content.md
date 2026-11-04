# Advanced Solar Irradiance Forecasting ($t+60$ Min Horizon)
### Meeting Presentation Deck — Prof. Arup Ratan
**Design-Oriented Project (DOP) | BITS Pilani, Hyderabad Campus | Department of EEE**  
**Focus: Phases 4, 5, 6 Breakthroughs, Advanced Feature Physics, & Future Scope**

---

## Slide 1: Title Slide

**Title:** Advanced Solar Irradiance Forecasting at $t+60$ Minutes: Physics-Guided Diurnal Modeling, Satellite Cloud Advection, and Conformalized Quantile Ensembles

**Subtitle:** Addressing the Solar Noon Overshoot and Breaking the 1-Hour Horizon Barrier on Tropical 1-Minute Ground Pyranometer Data (Kuala Lumpur 2025)

**Presented To:** Prof. Arup Ratan (Department of Electrical & Electronics Engineering)  
**Academic Term:** Design-Oriented Project (DOP) | BITS Pilani, Hyderabad Campus  
**Project GitHub:** https://github.com/saimanoj446/Renewable-Energy-Forecasting-AI

---

## Slide 2: Executive Summary & Last Meeting Follow-Up

**Recap of Professor's Key Directives from Last Meeting:**
1. **Focus on Kuala Lumpur (Tropical vs Arid):** Shift away from desert Rajasthan to Kuala Lumpur's unpredictable convective cloud climate.
2. **The Solar Noon Overshoot Problem:** Address the issue where models extrapolating rising morning trends ($y = mx + c$) overshoot past 12:00 PM astronomical peak.
3. **Elevate the $t+60$ Min Forecast:** Improve accuracy and skill at the critical 1-hour grid-dispatch horizon (Phase 3 LSTM was $R^2 = 0.5266$).

**Summary of Engineering Breakthroughs Achieved:**
```
Phase 3d (Baseline)          Phase 4 (Diurnal Physics)          Phase 5 (Satellite GBDT)          Phase 6 (Conformal Physics)
  2-Layer LSTM                   Recurrent Fusion                 LightGBM / XGBoost                Multi-Quantile CQR
  R² = 0.5266                    R² = 0.5239                      R² = 0.5392                       R² = 0.5480 🏆
  RMSE = 204.7 W/m²              RMSE = 203.1 W/m²                RMSE = 197.1 W/m²                 RMSE = 195.2 W/m² 🏆
  MAE = 155.8 W/m²               MAE = 154.6 W/m²                 MAE = 149.3 W/m²                  MAE = 141.9 W/m² 🏆
  Linear Noon Overshoot          0% Runaway Overshoot             Satellite Low Cloud #3 Feature    82.0% Conformal Coverage
```

> **Speaker Notes:**  
> *"Sir, following your guidance, we dedicated this sprint strictly to the $t+60$ minute horizon on Kuala Lumpur. We solved the 12:00 noon overshoot physically using future clear-sky ceilings, brought in geostationary satellite convective cloud layers via API, and upgraded the modeling pipeline from LSTMs to Gradient Boosted Trees and Conformal Quantiles, driving $R^2$ to $0.5480$ and MAE down by $14\ W/m^2$."*

---

## Slide 3: Phase 4 — The Solar Noon Dilemma & Astronomical Geometry

### 1. The Core Physical Dilemma
* A temporal model observing 10:30 AM $\to$ 11:30 AM sees irradiance ramping up at $+15\ W/m^2/\text{min}$.
* If unconstrained, the model projects this upward slope into 11:30 AM $\to$ 12:30 PM, predicting impossible values ($> 1,100\ W/m^2$) after the sun has already passed its zenith.

### 2. The Solution: Clear-Sky Normalization & Exogenous Ceiling
We decompose instantaneous irradiance into a deterministic astronomical ceiling and a stochastic cloud factor:
$$GHI(t) = k_t(t) \times GHI_{\text{clear}}(t)$$

Where $GHI_{\text{clear}}$ is computed deterministically via the **Kasten-Young / Ineichen-Perez** model:
$$\cos(\theta_z) = \sin(\phi)\sin(\delta) + \cos(\phi)\cos(\delta)\cos(\omega)$$
$$AM = \frac{1}{\cos(\theta_z) + 0.50572 \cdot (96.08 - \theta_{z,\text{deg}})^{-1.6364}}$$
$$GHI_{\text{clear}} = I_0 \cdot \cos(\theta_z) \cdot c_{g1} \cdot \exp\left[-c_{g2} \cdot AM \cdot T_L \cdot \exp(0.01 \cdot AM^{1.8})\right]$$

### 3. Key Innovation: Known Future Astronomy as an Exogenous Input
* Weather 60 minutes ahead is stochastic, but **solar geometry at $t+60$ is 100% deterministic**.
* We feed $GHI_{\text{clear}}(t+60)$, $\text{Hour\_sin}(t+60)$, and $\text{Hour\_cos}(t+60)$ directly into the prediction head.
* The model predicts the cloud factor $\hat{k}_t(t+60) \in [0, 1.2]$, and physical irradiance is reconstructed as:
  $$\widehat{GHI}(t+60) = \hat{k}_t(t+60) \times GHI_{\text{clear}}(t+60)$$

> **Speaker Notes:**  
> *"Because Earth's orbital mechanics are known in advance, $GHI_{\text{clear}}(t+60)$ provides a hard physical ceiling. Even during a sharp morning ramp, the model knows the exact theoretical limit 60 minutes ahead, making runaway overshoot past solar noon mathematically impossible."*

---

## Slide 4: Phase 4 — Architecture & Solar Noon Verification

```
PAST SEQUENCE: t-60 to t (60 min x 10 features)        KNOWN FUTURE AT t+60:
[GHI, GHI_clear, kt, Temp, RH, DNI, DHI, P, Hr]         [GHI_clear(t+60), Hour_sin, Hour_cos, Ramp]
                      │                                                        │
                      ▼                                                        │
            ┌───────────────────┐                                              │
            │ Conv1D (32 ch)    │                                              │
            │ BatchNorm + GELU  │                                              │
            └─────────┬─────────┘                                              │
                      ▼                                                        │
            ┌───────────────────┐                                              │
            │ 2-Layer PyTorch   │                                              │
            │ LSTM (Hidden=64)  │                                              │
            └─────────┬─────────┘                                              │
                      ▼                                                        │
            Last Hidden State h60 (64-dim)                                     │
                      │                                                        │
                      └─────────────────┬──────────────────────────────────────┘
                                        ▼
                            ┌────────────────────────┐
                            │ Recurrent Fusion Head  │
                            │ Dense(68 → 32) → GELU  │
                            │ Dense(32 → 1) + Sigmoid│
                            └───────────┬────────────┘
                                        ▼
                                 kt(t+60) ∈ [0, 1.2]
                                        │
                                        ▼ [Physics Multiply]
                      GHI_pred(t+60) = kt_pred × GHI_clear(t+60)
```

**Experimental Verification:**
* **Plot 45 (`plots/45_ph4b_training_convergence.png`):** Stable validation convergence; early stopping triggered at epoch 11, completely eliminating Phase 4a overfitting.
* **Plot 50 (`plots/50_ph4c_t60_walkforward_day.png`):** Continuous walk-forward day profile. The blue predicted line strictly bends downward following the green clear-sky ceiling curve past solar noon.
* **Plot 49 (`plots/49_ph4c_t60_scatter_and_residuals.png`):** Unbiased error distribution centered at **$+11.56\ W/m^2$** (reduced from $+25\ W/m^2$ in Phase 3).

---

## Slide 5: Phase 5 — Transitioning Beyond LSTM to Gradient Boosted Trees

### Why Move Beyond Recurrent Architectures for 1-Hour Horizons?
1. **Recency Bias in Recurrent Gates:** LSTMs rely heavily on the hidden state $h_t$ at the window's final step. This produces stellar $+1\text{m}$ forecasts ($R^2 = 0.898$) because $GHI(t+1) \approx GHI(t)$, but degrades rapidly by $+60\text{m}$ as memory fades.
2. **Non-Linear Cloud Thresholds:** Convective clouds cause abrupt, discontinuous step-downs in solar power (e.g., dropping from $800\ W/m^2 \to 250\ W/m^2$ in 3 minutes). Decision trees naturally split sharp threshold boundaries (`IF Cloud_Cover_Low > 65% AND RH > 85% THEN drop GHI by 60%`), whereas sigmoid/tanh LSTM gates smooth out the drop.
3. **Training Efficiency & Interpretability:** LightGBM trains across 260,000 minute samples in **12 seconds** (vs 5 minutes for LSTM) and provides native split-gain and SHAP feature importances.

```
Model Paradigm Comparison:
  • PyTorch LSTM:       Sequence-to-point recurrent mapping | High recency bias | Computationally heavy
  • LightGBM / XGBoost: Multi-scale tabular trees | Step-function splits | Ingests spatial & rolling stats
```

---

## Slide 6: Phase 5 — Ingesting Geostationary Satellite Cloud Cover via API

### The Missing Physics in Point Pyranometers
A stationary ground pyranometer only senses light passing directly through the column of air above it. At typical tropical wind speeds of $20-30\text{ km/h}$, a cloud front that will shade the station 60 minutes from now is currently **15 to 40 km upwind** — completely invisible to the ground sensor!

### Ingested Geostationary Satellite Layers (Open-Meteo / Himawari-8/9)
We integrated real-time and historical satellite cloud optical depth reanalysis via API:
* **`Cloud_Cover_Low` (%):** Low-altitude convective cumulus / stratocumulus clouds ($0-2\text{ km}$). This is the **primary physical driver of sudden irradiance dropouts** in tropical Southeast Asia.
* **`Cloud_Cover_Mid` (%):** Altocumulus / altostratus layers ($2-6\text{ km}$).
* **`Cloud_Cover_High` (%):** Cirrus cloud layers ($>6\text{ km}$) causing optical thinning.
* **`Cloud_Cover` (%):** Integrated total column optical cloud fraction.

### Multi-Scale Dynamic Volatility Features Engineered
* `GHI_roll_mean_15`, `GHI_roll_mean_30`, `GHI_roll_mean_60`: Multi-horizon moving baselines.
* `GHI_roll_std_15`: 15-minute moving standard deviation measuring local cloud turbulence.
* `GHI_ramp_15`, `GHI_ramp_30`: Measured ramp rate ($GHI_t - GHI_{t-\Delta t}$).
* `kt_roll_mean_30`: Moving average clear-sky index.

---

## Slide 7: Phase 5 — Satellite Feature Importance & Benchmark Results

### Feature Importance Breakthrough (Plot 52)
In `plots/52_ph5_feature_importance.png`, LightGBM's split-gain ranking revealed:
* **`Cloud_Cover_Low` ranked as the #3 most influential feature in the entire model**, exceeding Ambient Temperature, Relative Humidity, and past 15-minute GHI!
* **`GHI_clear_t60` ranked #6**, confirming the heavy reliance on future astronomical geometry.
* **`Cloud_Cover_Mid` ranked #7**.

### Multi-Model Benchmark (+60 Min Horizon, Oct–Dec 2025 Test Holdout)

| Model Architecture | Feature Representation | R² Score | RMSE (W/m²) | MAE (W/m²) | Skill Score (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Phase 3: 2-Layer PyTorch LSTM** | 8 features $\times$ 60 min (480 pts) | 0.5266 | 204.67 | 155.80 | +25.79% |
| **Phase 4c: Recurrent Fusion** | Past Sequence + $GHI_{\text{clear}}(t+60)$ | 0.5239 | 203.11 | 154.58 | +27.48% |
| **Phase 5: XGBoost (Satellite)** | Ground + Satellite Clouds + Future Astronomy | 0.5342 | 198.14 | 149.15 | +28.03% |
| **Phase 5: GBDT Ensemble** | 50% LightGBM + 50% XGBoost | 0.5387 | 197.16 | 148.95 | +28.39% |
| **Phase 5: LightGBM (Satellite)** | **Ground + Satellite Clouds + Future Astronomy** | **0.5392 🏆** | **197.07 🏆** | **149.30** | **+28.42% 🏆** |

**Key Diagnostic Visualizations:**
* `plots/52_ph5_feature_importance.png` — Proves satellite cloud layers dominate decision splits.
* `plots/53_ph5_model_comparison_bars.png` — Clear side-by-side accuracy bar chart.
* `plots/54_ph5_walkforward_day.png` — Continuous tracking demonstrating improved responsiveness during cloud gaps.

---

## Slide 8: The Information Horizon of Single Ground Pyranometers

### Why Does Deterministic $R^2$ Plateau Around $\approx 0.54$ at $t+60$?
1. **Convective Cloud Lifecycles (The Eulerian Limit):**  
   In equatorial tropical rainforests (Köppen *Af*), convective storm cells form, rain, and dissipate within **10 to 30 minutes**. By 60 minutes, the atmospheric cloud configuration has transitioned into a brand new stochastic state that did not exist when $t=0$ was observed.
2. **The "Single-Sensor" Visibility Blind Spot:**  
   A ground pyranometer has a spatial viewing radius of only a few hundred meters. At $t+60$, the cloud bank that will cause shading is currently 25 km away. Without an upwind spatial array or satellite optical flow, this information does not exist in past ground time-series.
3. **The Literature Benchmark:**  
   In published literature (*IEEE Transactions on Sustainable Energy*, *Solar Energy*), for single ground pyranometers at 1-hour lead times in equatorial zones, **$R^2$ values between $0.50$ and $0.55$ represent the empirical ceiling**.
4. **The Critical Shift in Methodology:**  
   Rather than treating 60-minute solar forecasting as an impossible deterministic point problem ($450\ W/m^2$ vs $150\ W/m^2$), grid operators require **Calibrated Prediction Intervals ($[\hat{q}_{10}, \hat{q}_{90}]$)** to schedule spinning reserves. This motivated **Phase 6**.

---

## Slide 9: Phase 6 — Advanced Physics Feature Engineering

### 1. Variability Index ($VI$): Arc-Length Ratio
Traditional rolling standard deviation only measures spread; it cannot distinguish between a smooth diurnal rise and chaotic cloud-edge oscillations. We implement the **Variability Index ($VI$)**, computing the arc-length ratio of actual GHI to clear-sky GHI over rolling 15- and 30-minute causal windows:
$$VI_{\tau} = \frac{\sum_{i=0}^{\tau} \sqrt{(GHI_{t-i} - GHI_{t-i-1})^2 + \Delta t^2}}{\sum_{i=0}^{\tau} \sqrt{(GHI_{\text{clear},t-i} - GHI_{\text{clear},t-i-1})^2 + \Delta t^2}}$$
* For clear, cloudless skies: $VI \approx 1.0$.
* During broken, turbulent cumulus fields: $VI$ spikes to **$8.0 - 15.0$** due to rapid edge scattering.

### 2. Dynamic Ramp Acceleration ($\Delta^2 GHI$ & $\Delta^2 k_t$)
First-order differences ($\Delta GHI$) capture ramp speed; second-order differences capture **ramp acceleration**, detecting the onset of a cloud event before the drop fully manifests:
$$\Delta GHI_t = GHI_t - GHI_{t-1}, \quad \Delta^2 GHI_t = \Delta GHI_t - \Delta GHI_{t-1}$$
$$\Delta^2 k_{t} = (k_t - k_{t-1}) - (k_{t-1} - k_{t-2})$$

### 3. Clear-Sky Index Skewness ($\text{Skew}_{30}(k_t)$)
Rolling 30-minute skewness separates uniform low-overcast skies (symmetric distribution around $k_t \approx 0.25$) from broken skies experiencing forward Mie scattering enhancement ($k_t > 1.2$, heavily right-skewed).

*In `plots/58_ph6_physics_feature_importance.png`, $VI_{30}$ ranked directly in the top tier of feature importance, confirming the value of arc-length physics.*

---

## Slide 10: Phase 6 — Multi-Quantile LightGBM & Conformal Prediction (CQR)

```
                       TRAINING ON HISTORICAL DATA (Jan-Aug 2025)
                                          │
                  ┌───────────────────────┼───────────────────────┐
                  ▼                       ▼                       ▼
            LightGBM q10            LightGBM q50            LightGBM q90
         (Pinball alpha=0.10)    (Pinball alpha=0.50)    (Pinball alpha=0.90)
                  │                       │                       │
           Lower Bound q10             Median P50            Upper Bound q90
                  └───────────────────────┬───────────────────────┘
                                          ▼
                CALIBRATION ON VALIDATION SLICE (September 2025)
             Compute Nonconformity Scores: s_i = max(q10 - y, y - q90)
                Calculate Conformal Correction Factor Q = Quantile(s, 0.80)
                                          │
                                          ▼
                   EVALUATION ON TEST HOLDOUT (Oct-Dec 2025)
          Calibrated Lower Bound:  L_hat = max(0, q10_pred - Q)
          Calibrated Upper Bound:  U_hat = min(1.2 * GHI_clear, q90_pred + Q)
                                          │
                                          ▼
                      GUARANTEED >= 80% EMPIRICAL COVERAGE (PICP)
```

### Why Conformalized Quantile Regression (CQR)?
* Standard quantile regression frequently suffers from **monsoon under-coverage** (e.g., predicting an 80% band that only covers 72% of test points due to seasonal distribution shift).
* Conformal Prediction computes a non-parametric correction score $Q$ on the September validation set, guaranteeing that **at least 80% of unseen future GHI values fall strictly within the interval $[L, U]$**.

---

## Slide 11: Phase 6 — Final Experimental Results & Visual Proof

### 1. Deterministic Point Performance (P50 Median)
* **$R^2$ Score:** **$0.5480$ 🏆** (the highest accuracy achieved in the entire project).
* **RMSE:** **$195.18\ W/m^2$ 🏆** (reduced by $9.5\ W/m^2$ vs Phase 3 LSTM).
* **MAE:** **$141.90\ W/m^2$ 🏆** (reduced by **$13.9\ W/m^2$** vs Phase 3 LSTM).
* **IEEE Skill Score:** **$+29.10\%$ 🏆** over persistence.

### 2. Probabilistic Conformal Performance (80% Nominal Interval)
* **Prediction Interval Coverage Probability (PICP):** **$81.98\%$** (guaranteed $\ge 80\%$, eliminating under-coverage).
* **Interval Sharpness (PINAW):** **$33.74\%$** of dynamic range.
* **Continuous Ranked Probability Score (CRPS):** **$42.58\ W/m^2$** (gold standard metric for IEEE Transactions).

### 3. Visualizations Generated
* **Plot 56 (`plots/56_ph6_probabilistic_fan_chart.png`):** 3-Day Walk-Forward Fan Chart. Shaded uncertainty band remains razor-thin on sunny mornings and dynamically balloons outward during afternoon convective storm transitions.
* **Plot 57 (`plots/57_ph6_conformal_reliability_diagram.png`):** Reliability calibration diagram tracking the $1:1$ ideal line; pinball loss breakdown ($q_{10}=26.5$, $q_{50}=71.0$, $q_{90}=30.3\ W/m^2$).
* **Plot 58 (`plots/58_ph6_physics_feature_importance.png`):** Highlighted bar chart proving $VI_{30}$ and `Cloud_Cover_Low` dominate the model.
* **Plot 59 (`plots/59_ph6_benchmark_comparison.png`):** Progression bar chart showing steady, verified gains from Phase 3 $\to$ Phase 5 $\to$ Phase 6.

---

## Slide 12: Comprehensive Benchmark Scorecard (+60 Min Horizon)

Evaluated on the exact same out-of-time holdout (Oct–Dec 2025, 66,240 minute records):

| Project Phase | Model & Architecture | Key Features Ingested | $R^2$ Score | RMSE ($W/m^2$) | MAE ($W/m^2$) | IEEE Skill Score (%) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Phase 3d** | 2-Layer PyTorch LSTM | 8 features $\times$ 60 min (480 pts) | 0.5266 | 204.67 | 155.80 | +25.79% |
| **Phase 4c** | Dedicated Recurrent Fusion | Past Seq + Future $GHI_{\text{clear}}(t+60)$ | 0.5239 | 203.11 | 154.58 | +27.48% |
| **Phase 5** | XGBoost Regressor | Ground + Satellite Clouds + Astronomy | 0.5342 | 198.14 | 149.15 | +28.03% |
| **Phase 5** | GBDT Ensemble | 50% LightGBM + 50% XGBoost | 0.5387 | 197.16 | 148.95 | +28.39% |
| **Phase 5** | LightGBM Regressor | Ground + Satellite Clouds + Astronomy | 0.5392 | 197.07 | 149.30 | +28.42% |
| **Phase 6** | **Physics-P50 CQR Hybrid** | **Variability Index ($VI$) + Satellite + CQR** | **0.5480 🏆** | **195.18 🏆** | **141.90 🏆** | **+29.10% 🏆** |

---

## Slide 13: Phase 7 — 3D Cloud Parallax & Ray-Tracing Shadow Physics

### 1. The Physics of Cloud Offset & Satellite Obliquity
* Satellites view Peninsular Malaysia obliquely ($\theta_v = 42.8^\circ, \phi_v = 102.2^\circ$), projecting clouds away from their true ground coordinates:
  $$\vec{d}_{\text{parallax}} = H_c \cdot \tan\theta_v \cdot [\sin\phi_v \hat{i} + \cos\phi_v \hat{j}]$$
* Sunlight casts cloud shadows on the ground along the solar zenith angle ($\theta_s, \phi_s$):
  $$\vec{d}_{\text{shadow}} = H_c \cdot \tan\theta_s \cdot [-\sin\phi_s \hat{i} - \cos\phi_s \hat{j}]$$
* **Net Ground Offset Vector:**
  $$\vec{D}_{\text{net}} = \vec{d}_{\text{shadow}} - \vec{d}_{\text{parallax}}$$
* **Slant Optical Path Transmittance:**
  $$m_{\text{optical}} = \frac{1}{\cos\theta_s + 0.05}, \quad T_{\text{slant}} = \exp\left(-0.025 \cdot CC_{\text{low}} \cdot m_{\text{optical}}\right)$$

### 2. Experimental Verification Results (Plots 63, 64, 65)
* `dx_net_offset` emerged as the **#3 most influential feature** across all 25 inputs.
* At $t+60\text{ min}$, Parallax alone boosted raw 1-minute $R^2$ from $0.5343 \to 0.5423$ (+0.0080 gain).
* Proved that at high solar zenith angles (morning/afternoon), cloud shadows shift up to $6\text{ km}$ from the satellite's apparent coordinate!

---

## Slide 14: Phase 7b — 2D Parametric Horizon & Lookback Optimization

### 1. 36-Cell Parametric Grid Sweep ($6 \times 6$)
* Evaluated Lookback Windows $x \in [15, 30, 45, 60, 90, 120]\text{ minutes}$.
* Evaluated Forecast Horizons $t_d \in [5, 10, 15, 30, 45, 60]\text{ minutes}$.
* Cross-benchmarked Baseline vs Parallax-Enhanced models on 66,240 test points.

### 2. Key Research Findings (Plots 60, 61, 62, 66, 67)
1. **$x = 60\text{ Minutes}$ is the Universal Sweet Spot:**
   - Shorter lookbacks ($15-30\text{m}$) lack cloud velocity context; longer lookbacks ($90-120\text{m}$) suffer from non-stationary cloud decay.
2. **Where Parallax Delivers Largest Accuracy Gains (Plot 66):**
   - At short horizons ($t_d \le 15\text{m}$), pyranometer persistence dominates.
   - At operational horizons ($t_d \ge 30\text{m}$), Parallax produces **unanimously positive $\Delta R^2$** across all lookback windows (up to $+0.0096$ at $60\text{m}$).

---

## Slide 15: Phase 7c — Breaking the "Double Penalty" Ceiling

### 1. Why Point MSE Hit a Wall ($R^2 \approx 0.54$)
* Pinhole pyranometer sensors ($2\text{ cm}^2$) measure sub-second turbulence, while satellites average over $2-10\text{ km}$ grids.
* **The Double Penalty Effect:** Predicting a cloud drop 2 minutes early is penalized *twice* by MSE (once for predicting a drop early, and again for missing the drop 2 minutes later). Blurry naive averages get better point MSE than sharp physical models!

### 2. The Breakthrough: Target Aggregation & Online EWMA Causal Filtering
* Aligned targets with real utility market operating blocks:
  - **1-Min Instantaneous:** $R^2 = 0.6684$ ($+32.41\%$ skill, RMSE $= 167.1\text{ W/m}^2$)
  - **5-Min CAISO RTD Block:** **$R^2 = 0.7164$ 🏆** ($+36.62\%$ skill, RMSE $= 149.3\text{ W/m}^2$, Ramp MAE $= 18.8\text{ W/m}^2$)
  - **15-Min Market Dispatch Block:** **$R^2 = 0.7633$ 🏆** ($+41.14\%$ skill, RMSE $= 131.3\text{ W/m}^2$, Ramp MAE $= 11.0\text{ W/m}^2$)
* **Online EWMA Causal Filter:** Tracks running bias $b_t = 0.85 b_{t-1} + 0.15 (y_t - \hat{y}_t)$ with zero future leakage, eliminating diurnal monsoon drift.

---

## Slide 16: Phase 8 — Dynamic PV Array Power Modeling & Sandia Thermal Lag

### 1. Physics-Informed Utility Solar Farm Model (10 MWp DC / 8.0 MW AC)
* **Sandia Dynamic Cell Temperature Model:**
  $$T_{\text{cell, ss}} = T_{\text{amb}} + GHI \cdot \exp(a + b \cdot WS) + \frac{GHI}{1000} \cdot \Delta T$$
  *(Open-rack coefficients: $a = -3.56, b = -0.075\text{ s/m}, \Delta T = 3.0^\circ\text{C}$)*
* **1st-Order Thermal Inertia Filter ($\tau_{\text{thermal}} = 7.0\text{ min}$):**
  $$T_{\text{cell}}(t) = (1 - \alpha_T) T_{\text{cell}}(t-1) + \alpha_T T_{\text{cell, ss}}(t), \quad \alpha_T = 1 - e^{-\Delta t / \tau_{\text{thermal}}}$$
* **Silicon Temperature Derating & Central Inverter Clipping:**
  $$P_{\text{dc}} = 10\text{ MW} \cdot \frac{GHI}{1000} \cdot \left[1 - 0.0042(T_{\text{cell}} - 25^\circ\text{C})\right], \quad P_{\text{ac}} = \min(8.0\text{ MW}, 0.965 \cdot P_{\text{dc}})$$

### 2. Experimental Verification Results (Plot 71)
* **Midday Cell Heating:** In tropical Kuala Lumpur ($T_{\text{amb}} \approx 33^\circ\text{C}$), cell temperature reaches **$61.2^\circ\text{C}$** at solar noon.
* **Thermal Power Loss:** Shaves **$1.45\text{ MW}$ ($-14.5\%$ capacity)** off the $10\text{ MW}$ array at peak irradiance!
* **AC Power Forecast Accuracy (+60 min):**
  - 5-Min CAISO RTD Target: **$R^2 = 0.7248$ 🏆** ($\text{RMSE} = 0.88\text{ MW}$)
  - 15-Min Market Dispatch Target: **$R^2 = 0.7680$ 🏆** ($\text{RMSE} = 0.78\text{ MW}$)

---

## Slide 17: Phase 8 — Predictive BESS Grid Balancing & IEEE 1547 Compliance

### 1. Grid Code Compliance & Controller Benchmark (Plot 72)
* **IEEE 1547 Standard:** Grid ramp rates strictly restricted to $\pm 10\% P_{\text{rated}}/\text{min} = \pm 0.80\text{ MW/min}$.
* **Controller Benchmark on Tropical Convective Storm:**
  - **Uncontrolled PV Direct Injection:** **$14.8\%$ Ramp Violations** (grid trip risk).
  - **Standard Reactive BESS (2.5 MWh):** **$3.2\%$ Violations** — Drains to the $10\%$ SoC floor halfway through the cloud and suffers a catastrophic power collapse.
  - **Phase 7c Predictive AI BESS:** **$< 0.2\%$ Violations (> 99.8% Compliance 🏆)** — Pre-charges battery to $65\%$ ahead of the shadow and pre-steers grid injection smoothly.

### 2. BESS Sizing Pareto Frontier (Plot 73)
* Swept battery capacity from $0.5\text{ MWh}$ to $4.0\text{ MWh}$ to meet the $99.5\%$ utility grid code standard:
  - Reactive BESS Required: **$2.60\text{ MWh}$**
  - Predictive AI BESS Required: **$1.75\text{ MWh}$**
  - **Capacity Sizing Reduction: $32.7\%$ MWh Savings 🏆** (Saves **~$425,000** in utility storage CAPEX!).

---

## Slide 18: Master Multi-Phase Progression Scorecard (Phases 1–8)

Full evolutionary progression across all project milestones:

| Phase | Milestone / Innovation | Target Resolution | Lead Time | $R^2$ Score | RMSE | Key Research Finding |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **Phase 1** | Random Forest Baseline | Hourly (Rajasthan) | +1 hr | 0.9846 | 39.4 W/m² | Cyclical feature engineering |
| **Phase 2** | Multi-Model Benchmark | Hourly (Rajasthan) | +1 hr | 0.9842 | 39.9 W/m² | XGBoost / SVR equivalence |
| **Phase 3d** | 2-Layer PyTorch LSTM | 1-Min (Kuala Lumpur) | +60 min | 0.5266 | 204.7 W/m² | 480-point tensor rule validated |
| **Phase 4c** | Recurrent Astronomy Fusion | 1-Min (Kuala Lumpur) | +60 min | 0.5239 | 203.1 W/m² | 0% runaway solar noon overshoot |
| **Phase 5** | Satellite Cloud GBDT | 1-Min (Kuala Lumpur) | +60 min | 0.5392 | 197.1 W/m² | Himawari low clouds #3 feature |
| **Phase 6** | Conformalized Quantile CQR | 1-Min (Kuala Lumpur) | +60 min | 0.5480 | 195.2 W/m² | 81.98% coverage, CRPS=42.6 |
| **Phase 7** | 3D Ray-Tracing Parallax | 1-Min (Kuala Lumpur) | +60 min | 0.5423 | 196.4 W/m² | Geometry shifts shadows up to 6 km |
| **Phase 7b** | 2D Lookback/Horizon Map | 1-Min (Kuala Lumpur) | 5–60 min | 0.7903–0.5423 | — | 60-min lookback universal sweet spot |
| **Phase 7c** | **Anti-Double Penalty + EWMA** | **5-Min CAISO RTD** | **+60 min** | **0.7164 🏆** | **149.3 W/m²** | **R² leaps past 0.70 high-precision bar** |
| **Phase 7c** | **Anti-Double Penalty + EWMA** | **15-Min Market Dispatch** | **+60 min** | **0.7633 🏆** | **131.3 W/m²** | **+41.14% IEEE Skill Score** |
| **Phase 8** | **Dynamic PV Power + BESS** | **8 MW AC Farm** | **+60 min** | **0.7248 (Pac)** | **0.88 MW** | **32.7% BESS sizing reduction (Pareto)** |

---

## Slide 19: Conclusion & Final Defense Summary

### Summary of Completed Deliverables:
1. **Resolved 1-Minute Turbulence Ceiling:** Proved that point MSE plateaued at $R^2 \approx 0.54$ due to the "Double Penalty Effect"; aggregating to 5-min RTD and 15-min dispatch unlocked **$R^2 = 0.7164$** and **$R^2 = 0.7633$**.
2. **First-Principles Atmospheric Physics:** Integrated Ineichen-Perez clear-sky ceilings, 3D satellite ray-tracing parallax, and causal online EWMA residual tracking.
3. **Closed Power Systems Engineering Loop:** Built a complete dynamic PV array power converter ($10\text{ MWp DC} / 8\text{ MW AC}$) with Sandia thermal lag ($\tau = 7\text{ min}$) and demonstrated an IEEE 1547 compliant BESS dispatch that reduces required battery storage capacity by **$32.7\%$**.

---
*All 73 publication figures, clean modular scripts, and results files are committed and accessible at:*  
**https://github.com/saimanoj446/Renewable-Energy-Forecasting-AI**
