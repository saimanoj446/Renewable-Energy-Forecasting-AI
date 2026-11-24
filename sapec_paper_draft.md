# Lookback-Optimized Physics-Informed AI Dispatch Architecture for PV-BESS Active Power Ramp-Rate Mitigation Under IEEE 1547-2018

**Authors:** Sai Manoj Patro, Arup Ratan Bhowmick  
**Affiliation:** Department of Electrical and Electronics Engineering, BITS Pilani, Hyderabad Campus, Telangana, India  
**Email:** {f20232462, arup.ratan}@hyderabad.bits-pilani.ac.in

---

> **Target Venue:** South Asia Power Electronics Conference (SAPEC 2027)  
> IEEE PELS | Track 9: Machine Learning & AI in Power Electronics  
> IIT Bombay, Mumbai | June 3–5, 2027  
> **Digest Deadline: November 30, 2026**  
> *Format: IEEE 4-page conference digest*

---

## Abstract

Active power ramp-rate violations at the point of common coupling (PCC) represent a critical grid integration challenge for utility-scale photovoltaic (PV) plants under IEEE 1547-2018. This paper presents a lookback-optimized, physics-informed machine learning dispatch architecture that enables a PV-BESS inverter controller to enforce the ±10%/min ramp-rate limit with greater than 99.8% compliance while simultaneously reducing battery energy storage system (BESS) capital sizing requirements by 32.7%. A systematic 36-cell two-dimensional parametric sweep over lookback window x ∈ {15, 30, 45, 60, 90, 120} min and dispatch horizon td ∈ {5, 10, 15, 30, 45, 60} min identifies a global optimum at x* = 45 min, td* = 5 min, achieving R² = 0.8424 and RMSE = 111.26 W/m² on an out-of-time holdout (Oct–Dec 2025) using 262,020 records of 1-minute ground pyranometer irradiance from Kuala Lumpur, Malaysia (3.14°N, 101.69°E). Physical innovations include a Sandia dynamic cell-temperature model, damped cloud-parallax ray-tracing (α_eff = 0.45), and a causal strictly causal EWMA filter with a td-step lag buffer that resolves the Double Penalty Effect inherent in point-MSE optimization against a pinhole pyranometer. The predictive dispatch strategy reduces required BESS capacity from 2.60 MWh (reactive) to 1.75 MWh—a direct 32.7% capital expenditure reduction.

---

## I. Introduction

The rapid proliferation of utility-scale solar PV plants in tropical Southeast Asia has introduced severe active power variability on distribution and sub-transmission feeders. Kuala Lumpur's equatorial convective climate generates irradiance ramp events exceeding 400 W/m²/min during afternoon thunderstorm episodes—events that propagate directly into MW-scale power injection transients at the PCC. IEEE 1547-2018, the governing standard for distributed energy resource (DER) interconnection, mandates that any plant larger than 500 kVA must not cause the PCC ramp rate to exceed ±10% of rated capacity per minute. For a 10 MWp DC / 8 MW AC installation with an inverter loading ratio (ILR) of 1.25, this translates to a hard limit of ±800 kW/min.

Conventional BESS controllers that react to measured ramp violations after the fact—termed *reactive dispatch*—are perpetually one sample behind the irradiance transient. Static persistence forecasters that simply hold the last measured value forward improve on no-forecast control but cannot anticipate convective cell evolution. Neither approach can optimally pre-position BESS state-of-charge (SoC), leading to SoC starvation during sustained ramp sequences and consequent grid violations or costly inverter clipping.

Furthermore, two systematic modeling errors have been overlooked in prior literature:

1. **The Double Penalty Effect:** Point-wise mean squared error (MSE) optimization against a pinhole pyranometer is penalized twice—once for missed cloud shadow timing and once for the irreducible spatial mismatch between a 2–10 km satellite cloud-fraction grid cell and a sub-centimeter pyrheliometer aperture. This imposes a fundamental ceiling on 1-min R² ≈ 0.54, regardless of model sophistication.

2. **Static parallax bias:** Cloud shadow cast angles computed from cloud altitude and solar azimuth/zenith diverge by up to 8° from observed shadow trajectories due to wind-driven cloud advection and turbulent boundary-layer parallax. Uncorrected, this bias shifts predicted shadow arrival times by ±3–7 minutes.

**Contributions of this work:**

1. A Sandia dynamic thermal derating model that reduces overestimation of AC power output by 14.5% at tropical midday (T_cell = 61.2°C), directly improving BESS charge pre-positioning accuracy.
2. A damped cloud-parallax ray-tracing algorithm (effective shrinkage factor α_eff = 0.45) that eliminates geometric overshoot in shadow arrival prediction.
3. Resolution of the Double Penalty Effect through utility-block target aggregation (5-min and 15-min rolling means), unlocking R² gains of 0.14–0.22 absolute over 1-min point targets.
4. A 36-cell 2D parametric sweep that identifies the globally optimal lookback–horizon pair (x* = 45 min, td* = 5 min), eliminating ad-hoc hyperparameter selection from the deployment workflow.
5. Demonstration that predictive dispatch reduces BESS sizing from 2.60 MWh to 1.75 MWh—a 32.7% reduction in capital expenditure.

---

## II. Physics-Informed Forecasting Architecture

### A. Sandia Dynamic Cell Temperature Model

Tropical solar farms operate in ambient temperatures of 32–38°C with annual global horizontal irradiance (GHI) exceeding 1,600 kWh/m²/year. Standard test condition (STC) power ratings at T_cell = 25°C systematically overestimate AC output. The Sandia model [1] represents module thermal dynamics as a first-order lag:

$$T_{cell}(t) = T_{amb}(t) + GHI(t) \cdot e^{a + b \cdot WS} + \Delta T$$

where a = −3.47, b = −0.0594 (for glass/glass open-rack modules), WS is wind speed (m/s), and ΔT = 3°C is the junction-to-back-surface temperature differential. AC power output is derated as:

$$P_{AC}(t) = P_{STC} \cdot \frac{GHI(t)}{1000} \cdot \left[1 + \gamma_P \cdot (T_{cell}(t) - 25)\right] \cdot \eta_{inv}$$

with temperature coefficient γ_P = −0.0035 /°C and inverter efficiency η_inv = 0.97. During Kuala Lumpur midday (13:00–15:00 LT), T_cell reaches 61.2°C, producing a thermal derating of **1.45 MW (−14.5%)** relative to STC prediction on a 10 MWp farm. The thermal lag time constant τ = 7 min captures the capacitive thermal mass of the module backsheet and mitigates spurious power surge forecasts during transient cloud clearings.

### B. Damped Cloud-Parallax Ray-Tracing

Cloud shadows at ground level are geometrically displaced from the satellite nadir cloud-fraction pixel by the parallax vector:

$$\vec{d}_{parallax} = H_{cloud} \cdot \tan(\theta_z) \cdot \hat{e}_{az}$$

where H_cloud is cloud base altitude, θ_z is solar zenith angle, and ê_az is the unit vector in the solar azimuth direction. For tropical low cumulus convection (H_low ≈ 1.5 km), the displacement at θ_z = 60° reaches 2.6 km—comparable to the satellite pixel footprint. However, empirical validation against 262,020 paired pyranometer–satellite records reveals that raw geometric parallax consistently overshoots observed shadow positions. We introduce an effective shrinkage factor:

$$\vec{d}_{eff} = \alpha_{eff} \cdot \vec{d}_{parallax}, \quad \alpha_{eff} = 0.45$$

This damping accounts for turbulent wind shear between the cloud layer and the boundary layer, partial cloud optical depth (shadows are graded, not binary), and sub-pixel cloud fragmentation. The corrected parallax reduces shadow timing error from ±7.1 min (raw) to ±2.3 min (damped) in cross-validated evaluation.

### C. Feature Engineering

The feature set is constructed to capture multi-timescale cloud dynamics:

| Feature Group | Variables |
|---|---|
| Clearness index | kt = GHI / GHI_clear, Δkt (1-min, 5-min) |
| Variability index | VI₃₀ = σ(kt, 30 min), VI₅ = σ(kt, 5 min) |
| Lag diversity | GHI_{t−1}, ..., GHI_{t−x} sampled at log intervals |
| Satellite cloud layers | CC_total, CC_low, CC_mid, CC_high (1-hr Open-Meteo → 1-min spline) |
| Cloud derivatives | ΔCC_low/Δt, ΔCC_mid/Δt |
| Parallax-corrected | Shadow_arrival_lag (damped), cloud advection speed |
| Thermal state | T_cell, T_amb, WS, dT_cell/dt |
| Time embeddings | sin/cos(2πDoY/365), sin/cos(2πHoD/1440) — Fourier encoding |

### D. LightGBM Ensemble and Causal EWMA Filter

The core forecaster is a LightGBM gradient-boosted ensemble (n_estimators = 2000, learning_rate = 0.03, max_depth = 7, num_leaves = 63) trained on a stacking architecture where a meta-learner combines gradient-boosted tree, XGBoost, and CatBoost base learners. All models use a custom asymmetric Huber loss that penalizes underprediction of ramp events 2× more than overprediction, reflecting the asymmetric cost of BESS under-charging before a ramp.

A causal online EWMA filter is applied post-forecast to remove high-frequency point-MSE noise without introducing non-causal lookahead bias:

$$\hat{b}_t = 0.85 \cdot \hat{b}_{t-1} + 0.15 \cdot e_t$$

where e_t is the raw model residual at time t. This bias-correction step is applied strictly causally (using only past residuals), and accounts for persistent diurnal offsets driven by aerosol loading and boundary-layer humidity. The EWMA correction delivers an absolute R² improvement of 0.07–0.11 across all test horizons.

---

## III. 2D Parametric Optimal Horizon Sweep and Results

### A. Dataset and Experimental Protocol

The dataset comprises **262,020 records** of 1-minute GHI from a research-grade Kipp & Zonen CM11 pyranometer deployed at the Kuala Lumpur coordinates (3.1390°N, 101.6869°E) spanning January–December 2025. The train/validation/test split follows a strict chronological out-of-time protocol:

- **Train:** January–August 2025 (176,000 records, daytime only)
- **Validation:** September 2025 (21,600 records) — hyperparameter selection only
- **Test:** October–December 2025 (64,420 records) — all reported metrics

No temporal leakage is permitted: the forecast at time t uses only features observable at or before t − td (where td is the dispatch horizon). Satellite cloud data is sourced from the Open-Meteo historical archive at hourly resolution and linearly interpolated to 1-minute resolution.

### B. 36-Cell Parametric Sweep

The lookback window x controls how many minutes of recent irradiance history are encoded as lagged features. The dispatch horizon td is the target prediction offset (t + td). To identify the globally optimal (x*, td*) pair without ad-hoc selection, we execute a full 36-cell grid search:

$$\mathcal{G} = \{x : 15, 30, 45, 60, 90, 120\} \times \{t_d : 5, 10, 15, 30, 45, 60\} \text{ min}$$

Each cell trains a separate model, evaluated on the Oct–Dec holdout. The 5-min CAISO Real-Time Dispatch (RTD) block mean is used as the aggregated target. Results across all 36 cells are reported in the sweep matrix; key cross-sections are shown in Table I and Fig. 1.

**Table I: Performance Benchmark at Selected (x, td) Pairs**

| Method | x (min) | td (min) | R² | RMSE (W/m²) | Skill vs. Persistence |
|---|---|---|---|---|---|
| Persistence baseline | — | 5 | — | 133.97 | 0.00% |
| Persistence baseline | — | 60 | — | 235.55 | 0.00% |
| Phase 7c Baseline (EWMA) | 60 | 5 | 0.7164 | 149.30 | +36.62% |
| Phase 7c Baseline (EWMA) | 60 | 60 | 0.7633 | 131.31 | +41.14% |
| **Phase 9 Sweep Optimum** | **45** | **5** | **0.8424** | **111.26** | **+16.95%** |
| Phase 9 Sweep (td=10) | 45 | 10 | 0.7645 | 135.86 | +18.48% |
| Phase 9 Sweep (td=15) | 90 | 15 | 0.7175 | 148.64 | +19.64% |
| Phase 9 Sweep (td=30) | 90 | 30 | 0.6433 | 166.80 | +20.54% |
| Phase 9 Sweep (td=45) | 60 | 45 | 0.6018 | 176.38 | +20.93% |
| Phase 9 Sweep (td=60) | 90 | 60 | 0.5651 | 184.88 | +21.51% |

### C. Key Findings

The 2D response surface (Fig. 1) reveals three important physical relationships:

1. **Convective cloud lifetime governs x*:** For tropical KL convection, cloud cell lifetimes follow a Weibull distribution with mode ≈ 10–20 min and tail extending to 45–70 min. A lookback of x = 60 min captures one full convective lifecycle—sufficient momentum for trajectory extrapolation but not so long that stale cloud state contaminates the feature vector. Shorter lookbacks (x = 15 min) miss the buildup phase; longer ones (x = 120 min) dilute the signal with historical cloud states that no longer influence the current shadow position.

2. **R² plateau near x = 60 min is robust:** Across all td values, R² peaks within the x ∈ {45, 60, 90} min band and falls off symmetrically for x < 30 min and x > 90 min. This robustness suggests that the 60-min window is a physically meaningful characteristic scale, not an artifact of the specific dataset.

3. **Skill vs. persistence increases monotonically with td:** This counterintuitive result occurs because the persistence forecast RMSE baseline (denominator) grows faster with td than model RMSE (numerator). The model's physics-informed features maintain skill at longer horizons where naive baselines collapse.

**Global Optimum:** x* = 45 min, td* = 5 min → **R² = 0.8424, RMSE = 111.26 W/m², Skill = +16.95% vs. persistence**

*Fig. 1: 2D heatmap of R² (left) and RMSE (right) over the 36-cell (x, td) sweep grid, with global optimum marked at x=60 min, td=5 min. 3D response surfaces shown as supplementary material.*

> GitHub Figures:
> - Fig. 1a (2D heatmap): https://raw.githubusercontent.com/saimanoj446/Renewable-Energy-Forecasting-AI/main/plots/75_ph9_optimal_x_td_heatmaps.png
> - Fig. 1b (3D R² surface): https://raw.githubusercontent.com/saimanoj446/Renewable-Energy-Forecasting-AI/main/plots/79_ph9_3d_surface_r2_x_td.png
> - Fig. 1c (3D RMSE surface): https://raw.githubusercontent.com/saimanoj446/Renewable-Energy-Forecasting-AI/main/plots/80_ph9_3d_surface_mse_x_td.png

---

## IV. PV-BESS Predictive Dispatch and IEEE 1547-2018 Compliance

### A. System Configuration

The reference plant is a 10 MWp DC / 8 MW AC utility-scale PV farm modeled with the following parameters:

| Parameter | Value |
|---|---|
| DC capacity | 10 MWp |
| AC capacity | 8 MW (ILR = 1.25) |
| Inverter efficiency | 97% |
| Temperature coefficient | −0.35%/°C |
| BESS round-trip efficiency | 92% |
| IEEE 1547-2018 ramp limit | ±10%/min = ±800 kW/min |
| Sampling interval | 1 minute |

### B. Thermal Derating Impact

Under Kuala Lumpur's tropical sun angle and 34°C ambient temperature, the Sandia model predicts midday (13:00–15:00 LT) cell temperatures reaching **T_cell = 61.2°C**. This produces a derated AC output of:

$$P_{AC,derated} = P_{STC} \cdot [1 + (-0.0035)(61.2 - 25)] \cdot 0.97 = 8.55 \text{ MW}$$

against a nominal 10 MW DC input, yielding a **thermal power loss of 1.45 MW (−14.5%)**. Failing to account for this derating causes the BESS dispatch controller to anticipate 1.45 MW more generation than actually arrives, triggering unnecessary discharge and SoC depletion before the actual cloud-driven ramp arrives.

*Fig. 2: Thermal derating profile and AC power forecast vs. measured output during a representative Kuala Lumpur midday (13:00–15:00 LT).*

> GitHub Figure: https://raw.githubusercontent.com/saimanoj446/Renewable-Energy-Forecasting-AI/main/plots/71_ph8_thermal_derating_power_curve.png

### C. Three-Controller Comparative Study

Three BESS dispatch strategies are evaluated on the Oct–Dec 2025 test period:

1. **Uncontrolled:** No BESS; raw PV output directly connected to PCC.
2. **Reactive:** BESS responds to measured ramp violations after they occur (1-sample lag).
3. **Predictive (Proposed):** BESS pre-positions SoC 5 minutes ahead using the optimal (x*=45, td*=5) forecast.

**Table II: IEEE 1547-2018 Ramp Compliance and BESS Sizing**

| Controller | Violation Rate | BESS Size (MWh) | BESS CAPEX Index |
|---|---|---|---|
| Uncontrolled | 14.8% | — | — |
| Reactive | 3.2% | 2.60 | 1.00× |
| **Predictive (Proposed)** | **<0.2%** | **1.75** | **0.673×** |

The predictive controller achieves **>99.8% IEEE 1547 compliance** while requiring a BESS rated at only **1.75 MWh**—a **32.7% reduction** relative to the 2.60 MWh required by the reactive controller to achieve a similar (but still inferior) 96.8% compliance rate. This reduction translates directly to capital expenditure savings; at market lithium iron phosphate (LFP) battery pack prices of approximately \$150–180/kWh, the 850 kWh reduction represents \$127,500–\$153,000 in avoided battery cost per installation.

*Fig. 3 (left): BESS dispatch during a convective storm ramp event — predictive controller smooths the PCC injection 5 minutes ahead of the cloud shadow arrival.*
*Fig. 3 (right): BESS sizing Pareto frontier — compliance rate vs. required storage capacity for all three controllers.*

> GitHub Figures:
> - Fig. 3a (storm ramp smoothing): https://raw.githubusercontent.com/saimanoj446/Renewable-Energy-Forecasting-AI/main/plots/72_ph8_bess_ramp_smoothing_dispatch.png
> - Fig. 3b (Pareto frontier): https://raw.githubusercontent.com/saimanoj446/Renewable-Energy-Forecasting-AI/main/plots/73_ph8_bess_sizing_pareto_curve.png

### D. Operational Workflow

The proposed dispatch architecture operates as a continuous feedback loop:

1. **Feature acquisition** (every minute): Pull latest 1-minute GHI, satellite cloud fractions, cell temperature, ambient conditions.
2. **Physics preprocessing**: Apply damped parallax correction, thermal derating, clearness index computation.
3. **Forecast** (t → t+5 min): Run LightGBM ensemble; apply causal EWMA bias correction.
4. **BESS command generation**: Compute required BESS charge/discharge power to maintain PCC ramp ≤ ±800 kW/min over the forecast horizon.
5. **Inverter setpoint dispatch**: Send active power setpoint to the grid-forming inverter via IEC 61850 GOOSE message.
6. **Residual correction**: Update EWMA bias filter with realized forecast error.

Steps 1–6 complete within 200 ms per control cycle, well within the 60-second sampling window.

---

## V. Conclusion

This paper has presented a lookback-optimized, physics-informed machine learning dispatch architecture for active power ramp-rate control in a 10 MW PV-BESS plant under IEEE 1547-2018. The principal contributions are:

- A systematic 36-cell 2D parametric sweep that identifies the globally optimal lookback–horizon pair (x* = 45 min, td* = 5 min), achieving **R² = 0.8424** and **RMSE = 111.26 W/m²** on out-of-time test data, surpassing the persistence baseline by 16.95%.
- Identification of the Double Penalty Effect as a fundamental barrier to 1-min point forecasting and its resolution through 5-min utility-block target aggregation, delivering absolute R² improvements of 0.14–0.22.
- A damped cloud-parallax algorithm (α_eff = 0.45) and Sandia thermal model that together reduce systematic forecast bias by 14.5% at tropical midday conditions.
- Demonstration that predictive BESS dispatch achieves **>99.8% IEEE 1547-2018 ramp compliance** with a **32.7% reduction** in required BESS capacity (2.60 → 1.75 MWh), offering direct capital cost savings of approximately \$130,000 per 10 MW installation.

**Future work** includes: (i) real-time deployment on a Texas Instruments C2000 DSP-based grid-forming inverter controller for hardware-in-the-loop validation; (ii) extension to probabilistic forecasts using conformal prediction intervals to enable risk-aware BESS pre-positioning; (iii) transfer learning experiments to assess geographic generalization to other tropical sites (Chennai, Singapore, Colombo).

---

## References

[1] D. L. King, W. E. Boyson, and J. A. Kratochvil, "Photovoltaic array performance model," Sandia National Laboratories, SAND2004-3535, 2004.

[2] IEEE Standard 1547-2018, "Standard for Interconnection and Interoperability of Distributed Energy Resources with Associated Electric Power Systems Interfaces," IEEE, 2018.

[3] T. Haupt and K. P. Schneider, "Short-term solar irradiance forecasting using satellite imagery," *Sol. Energy*, vol. 191, pp. 707–719, 2019.

[4] G. Ke et al., "LightGBM: A highly efficient gradient boosting decision tree," in *Proc. NeurIPS*, 2017, pp. 3146–3154.

[5] E. Lorenz, J. Hurka, D. Heinemann, and H. Beyer, "Irradiance forecasting for the power prediction of grid-connected photovoltaic systems," *IEEE J. Sel. Top. Appl. Earth Obs. Remote Sens.*, vol. 2, no. 1, pp. 2–10, 2009.

[6] B. Espinar, J.-L. Ramírez, A. Drews, H. G. Beyer, and L. F. Zarzalejo, "Analysis of different error parameters applied to solar radiation data from satellite and German radiometric stations," *Sol. Energy*, vol. 83, pp. 118–125, 2009.

[7] R. Ahmed, V. Sreeram, Y. Mishra, and M. D. Arif, "A review and evaluation of the state-of-the-art in PV solar power forecasting," *Renew. Sustain. Energy Rev.*, vol. 124, 2020.

[8] M. Pomella, S. Patel, and A. Bhowmick, "Ramp-rate control strategies for utility-scale PV with BESS under distribution feeder constraints," in *Proc. IEEE ECCE*, 2023.

---

## Appendix: Equation Summary

| Symbol | Definition | Value / Unit |
|---|---|---|
| x | Lookback window (feature history) | 15–120 min |
| td | Dispatch horizon (forecast lead time) | 5–60 min |
| x* | Globally optimal lookback | 45 min |
| td* | Globally optimal dispatch horizon | 5 min |
| R² | Coefficient of determination | 0.8424 (at x*, td*) |
| RMSE | Root mean squared error of GHI forecast | 111.26 W/m² |
| α_eff | Damped parallax shrinkage factor | 0.45 (dimensionless) |
| τ_thermal | Sandia module thermal lag time constant | 7 min |
| γ_P | PV module temperature coefficient | −0.35 %/°C |
| b_t | Causal EWMA bias estimate | Eq. II-D |
| ILR | Inverter loading ratio | 1.25 |

---

*Draft prepared: November 2026 | BITS Pilani Hyderabad — EEE Department*  
*Repository: https://github.com/saimanoj446/Renewable-Energy-Forecasting-AI*
