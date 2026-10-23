# GCEM Hacks 5.0 — Round 2 Review Presentation Deck
## Problem Understanding & Solution Design (50 Marks)
**Evaluation Date & Time:** 23/10/26 @ 09:00 PM  
**Project:** PhOTON-C²-UQ (Physics-informed Order-Transferred Optimal Network with Collaborative & Continual Learning, Conformal UQ & RUL Prognostics)  
**Team / Repository:** JerrySeba / GCEM_Hacks5.0  

---

### Slide 1: Problem Understanding & The Industrial Triad (20 Marks)
**Header:** Beyond Generic Audio ML: The Reality of Industrial Gearbox Monitoring  

* **The Problem Mandate:**
  * Gearboxes in wind turbines, aerospace powertrains, and electric drivetrains operate under dynamic speed/torque variations with **extreme failure data scarcity** (machines are healthy 99.9% of their life).
  * Must identify **Fault Type** (discrete isolation), **Fault Severity** (continuous damage %), **Remaining Useful Life (RUL)**, and **Prediction Confidence** across unseen machines.

* **The Industrial Triad (Why Standard DL Fails):**
  1. **Non-Stationary Speed Smearing:** Under fluctuating RPM, traditional Fourier/FFT and STFT cause severe Doppler-like spectral smearing, blending gear mesh frequencies ($f_{gm} = Z \cdot f_r$) with sidebands.
  2. **Extreme Failure Scarcity:** Real catastrophic failure data is rare, hazardous, and expensive. Training big models from scratch on failure samples is impossible.
  3. **Structural Domain Shift:** Sensor mounting, casing compliance, and transmission path transfer functions $H(\omega)$ vary drastically between machines (e.g., SEU Drivetrain rig vs. PHM 2009 multi-shaft gearbox).

* **Evaluation Benchmarks Used:**
  * **PHM 2009 NASA Challenge:** 30–50 Hz variable speed, high/low load, 3-axis vibration + optical tachometer.
  * **Southeast University (SEU) DDS:** Planetary + 2-stage parallel gearbox, 20–30 Hz, 8 synchronous channels.
  * **SpectraQuest / HUST Datasets:** Multi-speed, 0–90% variable torque loading.

---

### Slide 2: Innovation & Technical Architecture: PhOTON-C²-UQ (20 Marks)
**Header:** Hybrid Physics-Informed Machine Learning & Deep Domain Alignment  

```
 [Raw Acceleration x(t) + Tachometer] 
                   │
                   ▼ (1. Physics Pre-Processing)
 [Computed Order Tracking (COT) via Spline Resampling] ──► Removes RPM Doppler Smearing
                   │
                   ▼ (2. Deep Representation Backbone)
 [SE-ResNet-18 (Squeeze-and-Excitation Residual Network)] ──► Inter-channel Harmonic Re-weighting
                   │
         ┌─────────┴─────────┐
         ▼                   ▼
 [Deep CORAL Alignment]  [Latent Representation z]
 (Covariance Matching        │
  Across Machines)           ├──► Head 1: Evidential Dirichlet (Epistemic Vacuity u in [0, 1])
                             ├──► Head 2: Continuous Severity Estimator (Anchored on SER)
                             └──► Head 3: Physics-Informed Fatigue RUL (Paris-Erdogan ODE)
```

* **Core Breakthroughs:**
  1. **Computed Order Tracking (COT):** Uses cubic spline interpolation on keyphasor pulses to resample non-stationary time $x(t)$ into stationary angular domain $x(\theta)$, pinning gear mesh orders to fixed integer positions regardless of speed swings.
  2. **Deep CORAL Domain Adaptation:** Minimizes Frobenius norm distance between second-order statistics (covariance matrices) of source (SEU) and target (PHM 2009) latent representations, achieving cross-machine transfer with minimal target labels.
  3. **Evidential Deep Learning (EDL):** Replaces heuristic Softmax with a Dirichlet distribution parameterization, outputting explicit **Epistemic Vacuity ($u$)**. When an unseen or out-of-distribution machine vibration is fed, $u \to 1.0$ instead of overconfident hallucinations.

---

### Slide 3: Collaborative & Continual Learning Architecture (10 Marks)
**Header:** Multi-Rig Fleet Intelligence Without Raw Data Centralization  

* **Collaborative Learning: Federated Prototypical Distillation (FedProto)**
  * **Zero Raw Data Transfer:** Multiple industrial test rigs (Node A: SEU Rig, Node B: PHM 2009 Rig) do not share raw gigabyte vibration recordings (protecting bandwidth and proprietary telemetry).
  * **Latent Prototype Exchange:** Nodes exchange only compact mean class prototype vectors $C_k = \frac{1}{|S_k|} \sum_{i} f_\theta(x_i)$.
  * **Local Projection Loss:** Models align local feature extractors to global shared prototypes:
    $$\mathcal{L}_{\text{proto}} = \| f_\theta(x) - C_{y} \|_2^2$$
  * **Result:** Enables cross-machine generalization even when individual rigs have only 1–3 failure instances.

* **Continual Learning: Elastic Weight Consolidation (EWC)**
  * **The Catastrophic Forgetting Trap:** Fine-tuning on a new machine usually erases previously learned failure patterns from earlier machines.
  * **Quadratic Fisher Information Penalty:**
    $$\mathcal{L}_{\text{EWC}}(\theta) = \mathcal{L}_{\text{new}}(\theta) + \sum_{i} \frac{\lambda}{2} F_i (\theta_i - \theta_{A, i}^*)^2$$
  * Freezes parameters critical to previously mastered operating regimes while allowing plastic parameters to adapt to new gear ratios.

---

### Slide 4: Prognostic RUL & Conformal Uncertainty Bounds (10 Marks)
**Header:** Actionable Maintenance Windows with Mathematically Guaranteed Coverage  

* **Physics-Informed RUL via Paris-Erdogan Fatigue Mechanics:**
  * Gear tooth root cracks propagate under cyclic bending stress $\Delta \sigma$:
    $$\frac{da}{dN} = C (\Delta K)^m = C \left(Y \cdot \Delta \sigma \sqrt{\pi a}\right)^m$$
  * Integrating to critical shear threshold $a_{\text{crit}}$ yields remaining fatigue cycles:
    $$N_{\text{RUL}} = \int_{a(t)}^{a_{\text{crit}}} \frac{da}{C (Y \Delta \sigma \sqrt{\pi a})^m}$$
  * Dynamically adjusted in real time as operating load factor (%) varies.

* **Split-Conformal Calibration (Distribution-Free 95% Coverage):**
  * Rather than heuristic confidence scores, we compute non-conformity quantiles on calibration data:
    $$P\big(y \in [\hat{y} - \hat{q}, \; \hat{y} + \hat{q}]\big) \ge 1 - \alpha \quad (\alpha = 0.05)$$
  * Guarantees that continuous damage severity and RUL fall within the predicted interval 95% of the time, regardless of underlying data distribution.

* **Deployment Feasibility (100% Offline Edge SCADA):**
  * Lightweight footprint (~35 MB PyTorch model, < 10 ms inference latency).
  * Served locally via interactive Streamlit SCADA dashboard on `localhost:8501`, fully immune to venue Wi-Fi failures during judging.

---

## 🎙️ 90-Second Judge Elevator Pitch & Defense Q&A

### The 90-Second Delivery Script:
> "Good evening, judges. We are presenting **PhOTON-C²-UQ** for Gearbox Fault Diagnosis under Variable Conditions with Limited Failure Data.
> 
> Most conventional AI models fail in rotating machinery because they feed raw time-series or standard FFTs into neural networks. When an industrial gearbox changes RPM, FFT frequencies smear across bins, and when tested on an unseen gearbox, casing transfer functions cause models to misclassify completely.
> 
> We solved this through three core innovations:
> First, **Physics-Informed Computed Order Tracking (COT)** demodulates optical tachometer pulses and resamples the signal into the angular domain, making gear mesh orders completely invariant to RPM swings.
> 
> Second, our **Deep CORAL and Prototypical domain adaptation** aligns covariance manifolds between disparate rigs—such as the Southeast University DDS and PHM 2009—so that our model generalizes with as few as 2 labeled failure samples.
> 
> Third, we don't output unverified guesses: our **Evidential Dirichlet head** quantifies epistemic uncertainty, our **Paris-Erdogan fatigue law** projects remaining useful life, and **Split-Conformal Prediction** provides a mathematically guaranteed 95% confidence interval on both severity and RUL.
> 
> Our model trains in Google Colab and runs 100% offline on our laptop via our interactive SCADA dashboard. Thank you, and we'd love to take your questions!"

### Anticipated Judge Questions & Instant Winning Responses:
1. **Q: Why not use a 100% pure physics dynamic model (PINN / 8-DOF lumped parameter)?**
   * *A:* "Pure analytical lumped-parameter ODE solvers like Radau or LSODA suffer from numerical stiffness and divergence when contact backlash discontinuities occur under variable load. Furthermore, real physical casings have unmodelled transmission transfer functions. Our hybrid approach uses physics where it is exact (Computed Order Tracking and Paris' crack growth laws) while using deep metric learning to capture complex casing dynamics."

2. **Q: How does your model know if an input is something it hasn't seen before?**
   * *A:* "Instead of a standard Softmax that forces probabilities to sum to 1.0 even on gibberish, our Evidential Dirichlet head outputs belief masses and epistemic vacuity $u = K / \sum \alpha_k$. When given sensor noise or an unknown failure mode, $u \to 1.0$, immediately flagging an Out-of-Distribution SCADA alarm."

3. **Q: How do you validate cross-machine performance?**
   * *A:* "We train on the Southeast University DDS dataset and evaluate zero-shot and 2-shot transfer onto the NASA PHM 2009 dataset across 5 different shaft speeds and multiple load levels, using Hamming distance and conformal coverage metrics."
