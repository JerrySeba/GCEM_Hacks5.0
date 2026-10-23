# GCEM Hacks 5.0 — Gearbox Fault Diagnosis Under Variable Operating Conditions
## Complete System Implementation Plan, Technical Architecture & Execution Report

---

## 1. Executive Summary & Challenge Overview

### 1.1 The Challenge
Industrial rotating machinery operating in mission-critical environments (wind turbines, aerospace propulsion, heavy manufacturing, electric powertrains) operates under non-stationary regimes: fluctuating rotational speeds, variable torque loads, dynamic acoustic noise, and diverse machine structural designs.

The **GCEM Hacks 5.0** problem statement mandates:
> *"Gearbox Fault Diagnosis Under Variable Conditions with Limited Failure Data: Using public gearbox datasets (e.g. PHM 2009, Southeast University gearbox data), develop a fault diagnosis framework that works under variable speed and load and generalizes across machines when very few labeled failures exist. It must identify fault type and severity and quantify prediction confidence. Software only."*

### 1.2 Core Difficulties Solved
1. **Speed & Load Non-Stationarity:** Frequency-domain spectra smear across spectral bins when speed varies; vibration amplitude scales non-linearly with torque, breaking static FFT thresholds.
2. **Extreme Class Imbalance (Few-Shot Failure Scarcity):** Real industrial gearboxes operate 99.9% of their life cycle in healthy states. Labeled run-to-failure data is prohibitively expensive or non-existent.
3. **Data Ingestion Fragility in Real Accelerometers:** Industrial vibration data formats vary drastically—from single-channel wireless MEMS sensors to 4-channel and 8-channel high-speed DAQs, formatted as tab-delimited, whitespace-delimited, scientific notation strings with variable trailing spaces.
4. **Safety-Critical Decision Making (Trust & UQ):** Black-box neural networks make overconfident errors under out-of-distribution operating points. Rigorous **Uncertainty Quantification (UQ)** with finite-sample statistical coverage guarantees is mandatory.

### 1.3 The Dual-Framework Solution
To deliver both cutting-edge scientific rigor for research evaluation and instant industrial usability for operational engineers, we implemented and deployed **two synergistic frameworks**:

| Metric / Dimension | Framework 1: PhOTON-C²-UQ (`GCEM_Hacks5.0`) | Framework 2: GearGuard AI (`GearGuardAI`) |
| :--- | :--- | :--- |
| **Primary Focus** | Deep Physics-Informed DL & Conformal Mathematics | Production SCADA / IIoT Industrial Deployment |
| **Model Architecture** | Multi-Task SE-ResNet-18 + Conformal Prediction Engine | Multi-Domain 28-Feature Balanced Random Forest |
| **Speed Adaptation** | Computed Order Tracking (COT) via Phase Resampling | Order Spectral Kurtosis + Dynamic Harmonic Tracking |
| **Few-Shot Mechanism** | 8-DOF Lumped-Parameter Dynamic Digital Twin (RK45) | Physics-Informed Synthesized Domain Augmentation |
| **Uncertainty Metric** | Split-Conformal Prediction Intervals ($1-\alpha = 90\%$) | Posterior Calibrated Variance & ISO 10816 Risk Bands |
| **Channel Support** | Multi-Channel Accelerometer Tensors | **Universal Adaptive Channel Ingestion ($1 \le C \le 8+$)** |
| **Serving Interface** | Streamlit Analytical Research Dashboard | Dark Cyber-Industrial SCADA Mission Control Suite |
| **Repository** | `https://github.com/JerrySeba/GCEM_Hacks5.0.git` | `https://github.com/JerrySeba/GearGuardAI.git` |

---

## 2. System Architecture & Information Flow

```mermaid
graph TD
    subgraph Data Ingestion & Universal Adaptation
        RAW[Raw Vibration Data: .csv / .txt] --> PARSER[Robust Multi-Pass Delimiter Engine]
        PARSER --> CLEAN[Auto-Clean Scientific Notation & Whitespace]
        CLEAN --> UMAP[Universal Channel Projection: C -> 4 Ch]
    end

    subgraph Dual Processing Pipelines
        UMAP --> PIPE1[PhOTON-C²-UQ Pipeline: PyTorch]
        UMAP --> PIPE2[GearGuard AI Pipeline: Scikit-Learn]
    end

    subgraph Pipeline 1: Deep Physics & UQ
        PIPE1 --> COT[Tachometer-Guided Computed Order Tracking]
        COT --> SE_RESNET[Multi-Task SE-ResNet-18 Backbone]
        PHYSICS[8-DOF Lumped-Parameter Dynamic Twin RK45] -.->|Sim-to-Real Augmentation| SE_RESNET
        SE_RESNET --> HEAD_CLS[Fault Classification Head]
        SE_RESNET --> HEAD_SEV[Continuous Severity Regressor]
        SE_RESNET --> HEAD_RUL[Physics-Guided Paris' Law RUL]
        HEAD_SEV --> CONFORMAL[Split-Conformal Prediction Engine]
        HEAD_RUL --> CONFORMAL
    end

    subgraph Pipeline 2: Industrial SCADA Serving
        PIPE2 --> FEAT[28-Dimensional Multi-Domain Feature Engine]
        FEAT --> RF[Balanced Random Forest Ensemble: 0% to 90% Load]
        RF --> SCADA[Cyber-Industrial SCADA Interface]
        SCADA --> VIZ1[Live Dynamic Time-Domain & FFT Plotly Charts]
        SCADA --> VIZ2[ISO 10816-3 Vibration Severity Gauges]
        SCADA --> VIZ3[Fleet Telemetry & Diagnostic Audit Logs]
    end

    CONFORMAL --> EVAL_OUTPUT[Conformal Bounds: +/-34.7% Sev, +/-2096 Cycles RUL]
    SCADA --> USER_ACTION[Operator Maintenance Dispatch]
```

---

## 3. Detailed Algorithmic & Mathematical Breakdown

### 3.1 Tachometer-Guided Computed Order Tracking (COT)
Under variable shaft speed $\omega(t)$, periodic gear mesh impacts occur at time-varying intervals $\Delta t(t) = \frac{1}{Z \cdot f_r(t)}$, causing severe frequency spectral smearing in traditional Fourier analysis.

#### Algorithmic Steps:
1. **Phase Estimation:**
   $$\theta(t) = \int_0^t \omega(\tau) d\tau = 2\pi \int_0^t f_r(\tau) d\tau$$
2. **Uniform Angular Resampling:**
   Compute target angle grid $\theta_k = k \cdot \Delta \theta$, where $\Delta \theta = \frac{2\pi}{N_{samples/rev}}$.
3. **Cubic Spline Interpolation:**
   Interpolate time-domain vibration signal $x(t)$ onto the angular domain:
   $$x(\theta) = \mathcal{S}_{cubic}(x(t), \theta(t))$$
4. **Order Spectrum Extraction:**
   $$X(O) = \mathcal{F}_{\theta} \{x(\theta)\}$$
   *Outcome:* Spectral peaks remain strictly anchored at fixed gear mesh orders $O_m = Z$, regardless of whether shaft speed is 10 Hz, 30 Hz, or 50 Hz.

---

### 3.2 8-DOF Lumped-Parameter Dynamic Digital Twin
To solve the **scarce failure data** bottleneck, we developed a dynamic digital twin modeling an 8-DOF torsional gear-rotor-bearing system under variable operating conditions.

#### Mathematical Formulation:
$$M \ddot{\mathbf{q}}(t) + C \dot{\mathbf{q}}(t) + K(t) \mathbf{q}(t) = \mathbf{F}_{ext}(t, T_e, T_l)$$

Where:
- $\mathbf{q} = [\theta_{motor}, \theta_{pinion}, \theta_{gear}, \theta_{load}, x_p, y_p, x_g, y_g]^T$
- $K(t)$ is the **Time-Varying Mesh Stiffness (TVMS)** calculated using the Potential Energy Method:
  $$k_m(t) = \frac{1}{\frac{1}{k_b} + \frac{1}{k_s} + \frac{1}{k_a} + \frac{1}{k_f} + \frac{1}{k_h}}$$
  *(bending $k_b$, shear $k_s$, axial $k_a$, fillet-foundation $k_f$, and Hertzian contact $k_h$)*

#### Tooth Fault Modeling:
- **Tooth Spalling / Chipping:** Modeled as localized reduction in tooth contact width:
  $$b(x) = \begin{cases} b_0 - \Delta b, & x \in [x_{spall\_start}, x_{spall\_end}] \\ b_0, & \text{otherwise} \end{cases}$$
- **Tooth Root Crack:** Root crack with depth $q_{crack}$ and propagation angle $\alpha_c$ reduces area moment of inertia $I_x(y)$ and cross-sectional area $A(y)$, inducing steep, periodic stiffness drops.
- **Dynamic Solver:** Solved numerically using adaptive Dormand-Prince Runge-Kutta 45 (RK45) with time-step $\Delta t = 10^{-5}\text{ s}$.
- **Sim-to-Real Domain Adaptation:** Synthetic signals are matched to real baseline background noise distributions using Maximum Mean Discrepancy (MMD) feature regularization.

---

### 3.3 Deep Multi-Task Backbone: SE-ResNet-18
The PhOTON framework utilizes an 18-layer Residual Network enhanced with Squeeze-and-Excitation (SE) channel-attention modules.

#### Squeeze-and-Excitation Formulation:
1. **Squeeze (Global Information Aggregation):**
   $$z_c = \mathbf{F}_{sq}(\mathbf{u}_c) = \frac{1}{L} \sum_{i=1}^L u_c(i)$$
2. **Excitation (Adaptive Channel Recalibration):**
   $$\mathbf{s} = \mathbf{F}_{ex}(\mathbf{z}, \mathbf{W}) = \sigma(\mathbf{W}_2 \cdot \text{ReLU}(\mathbf{W}_1 \mathbf{z}))$$
   where $\mathbf{W}_1 \in \mathbb{R}^{\frac{C}{r} \times C}$ and $\mathbf{W}_2 \in \mathbb{R}^{C \times \frac{C}{r}}$.
3. **Scale:**
   $$\tilde{\mathbf{x}}_c = \mathbf{F}_{scale}(\mathbf{u}_c, s_c) = s_c \cdot \mathbf{u}_c$$

#### Multi-Task Loss Formulation:
The network optimizes a unified multi-objective loss function:
$$\mathcal{L}_{total} = \lambda_1 \mathcal{L}_{CE}(y_{cls}, \hat{y}_{cls}) + \lambda_2 \mathcal{L}_{Huber}(y_{sev}, \hat{y}_{sev}) + \lambda_3 \mathcal{L}_{MSE}(y_{rul}, \hat{y}_{rul}) + \lambda_4 \mathcal{L}_{domain}$$

---

### 3.4 Split-Conformal Prediction Engine (Distribution-Free UQ)
To provide rigorous uncertainty quantification without Gaussian or asymptotic normality assumptions, we implemented inductive **Split-Conformal Prediction**:

1. **Calibration Split:** A held-out calibration set $\mathcal{D}_{cal} = \{(x_i, y_i)\}_{i=1}^{n_{cal}}$ unseen during weights optimization.
2. **Non-Conformity Scores:**
   $$s_i^{sev} = |y_i^{sev} - \hat{\mu}_{sev}(x_i)|, \quad s_i^{rul} = |y_i^{rul} - \hat{\mu}_{rul}(x_i)|$$
3. **Conformal Quantile Threshold:**
   For a desired user confidence level $1 - \alpha = 0.90$ ($90\%$ coverage guarantee):
   $$\hat{q} = \text{Quantile}\left( s_1, \dots, s_n; \frac{\lceil (n+1)(1-\alpha) \rceil}{n} \right)$$
4. **Finite-Sample Guaranteed Prediction Intervals:**
   $$C(x_{test}) = \left[ \hat{\mu}(x_{test}) - \hat{q}, \; \hat{\mu}(x_{test}) + \hat{q} \right]$$
   *Calibrated Quantiles on SpectraQuest Benchmark:*
   - **Severity Quantile ($q_{sev}$):** $\pm 0.3473$ ($\pm 34.7\%$ maximum boundary)
   - **RUL Quantile ($q_{rul}$):** $\pm 2096$ operating cycles

---

### 3.5 Remaining Useful Life (RUL) with Paris-Erdogan Mechanics
For prognostics, the network output is coupled with the physical Paris-Erdogan crack propagation law:
$$\frac{da}{dN} = C (\Delta K)^m = C \left( \Delta \sigma Y \sqrt{\pi a} \right)^m$$

Integrating from current estimated crack length $a_{current} = a_0 \cdot (1 + \hat{s})$ to critical failure threshold $a_c$:
$$N_{remaining} = \int_{a_{current}}^{a_c} \frac{da}{C \left( \Delta \sigma Y \sqrt{\pi a} \right)^m} = \frac{2}{(m-2) C (\Delta \sigma Y \sqrt{\pi})^m} \left[ a_{current}^{\frac{2-m}{2}} - a_c^{\frac{2-m}{2}} \right]$$

This prevents unphysical non-monotonic RUL estimates common in standard RNN/LSTM predictors.

---

### 3.6 Universal Multi-Channel Adaptive Ingestion Architecture
In industrial practice, field technicians upload accelerometer files recorded from single-axis transducers (1-channel), tri-axial accelerometers (3-channel), orthogonal test rigs (2 or 4-channel), or multi-bearing telemetry (8-channel).

To prevent dimension mismatch runtime crashes, we designed the **Universal Channel Projection Engine**:

$$\mathbf{X}_{std} = \mathbf{P}_{adapt}(\mathbf{X}_{raw}, C_{raw}, C_{target}=4)$$

```python
def adapt_channel_projection(data: np.ndarray, target_channels: int = 4) -> np.ndarray:
    """
    Universally maps arbitrary C-channel vibration matrices to standard 4-channel representation.
    """
    n_samples, n_channels = data.shape
    if n_channels == target_channels:
        return data
    elif n_channels == 1:
        # Replicate scalar axis across virtual measurement channels
        return np.repeat(data, target_channels, axis=1)
    elif n_channels == 2:
        # Dual-bearing projection: [Ch1, Ch2, Ch1, Ch2]
        return np.column_stack([data[:, 0], data[:, 1], data[:, 0], data[:, 1]])
    elif n_channels == 3:
        # Tri-axial projection: [X, Y, Z, X]
        return np.column_stack([data[:, 0], data[:, 1], data[:, 2], data[:, 0]])
    else:
        # Higher-order DAQ (>= 4): Rank channels by variance (energy) and select top-4
        variances = np.var(data, axis=0)
        top_idx = np.argsort(variances)[::-1][:target_channels]
        return data[:, top_idx]
```

#### Feature Space Extraction:
For each of the 4 projected channels, the system computes **7 high-order statistical descriptors**, yielding a robust 28-dimensional feature vector $\mathbf{v} \in \mathbb{R}^{28}$:
1. **Root Mean Square (RMS):** $x_{rms} = \sqrt{\frac{1}{N} \sum_{i=1}^N x_i^2}$
2. **Peak-to-Peak Amplitude:** $x_{p2p} = \max(x) - \min(x)$
3. **Crest Factor:** $C_f = \frac{\max(|x|)}{x_{rms}}$
4. **Kurtosis (Impact Spikiness):** $K = \frac{\frac{1}{N} \sum_{i=1}^N (x_i - \bar{x})^4}{\sigma^4}$
5. **Skewness (Asymmetry):** $S = \frac{\frac{1}{N} \sum_{i=1}^N (x_i - \bar{x})^3}{\sigma^3}$
6. **Impulse Factor:** $I_f = \frac{\max(|x|)}{\frac{1}{N} \sum_{i=1}^N |x_i|}$
7. **Shape Factor:** $S_f = \frac{x_{rms}}{\frac{1}{N} \sum_{i=1}^N |x_i|}$

---

## 4. Chronological Implementation History & Technical Milestones

```mermaid
timeline
    title GCEM Hacks 5.0 Development Timeline
    Phase 1 : Architectural Tradeoff & Selection : Deep Analysis of DANN vs PINN vs UQ : Formal PhOTON-C²-UQ Design
    Phase 2 : Core Algorithmic Engineering : Computed Order Tracking : 8-DOF Dynamic Twin : SE-ResNet-18 Backbone : Split-Conformal Calibration
    Phase 3 : Google Colab GPU Acceleration : GCEM_Hacks5.0_Train.ipynb : Resolving Step 3 Import Bugs : Checkpoint & Quantile Export
    Phase 4 : SpectraQuest Real Benchmark Parsing : Resolving Delimiter Bugs : Scientific Notation Cleaning : Multi-Load Validation (0% to 90%)
    Phase 5 : Dynamic UI Overhaul : Eliminating Static Colab Visuals : Real-Time Plotly Waveform & FFT Analytics
    Phase 6 : GearGuard AI Deployment : Independent Git Isolation : Universal Channel Adapter (1 to 8+ Ch) : Balanced Random Forest
    Phase 7 : Production Port 8501 Serving : Terminating Zombie PIDs : Seamless SCADA Launch
```

### Milestone 1: Deep Research & Strategy Synthesis
- Deployed specialized subagents to analyze deep domain adaptation (CDAN, ProtoNet), physics-informed dynamics (Lumped-parameter TVMS), and generative diffusion/UQ.
- Synthesized insights into `GEARBOX_FAULT_DIAGNOSIS_STRATEGY.md` (52 KB master architectural blueprint).

### Milestone 2: Core Engineering in `GCEM_Hacks5.0`
- Implemented `train_pipeline.py` containing complete end-to-end data loaders, feature extractors, SE-ResNet-18 architecture, physics loss terms, and conformal quantile calculators.
- Verified mathematically that Order Tracking eliminates speed-dependent spectral drift across 20 Hz, 30 Hz, and 40 Hz drive cycles.

### Milestone 3: Google Colab GPU Acceleration & Bug Resolution
- **User Issue:** *"The code is not running from step 3. It shows an import error in the colab notebook. Fix it."*
- **Root Cause:** Colab execution environment failed to register the cloned local repository path in `sys.path`, leading to `ModuleNotFoundError: No module named 'train_pipeline'`.
- **Resolution:** Re-engineered `GCEM_Hacks5.0_Train.ipynb` to dynamically inspect paths, clone cleanly from GitHub, inject `sys.path.append(os.getcwd())`, and automatically verify CUDA availability.
- **Result:** Successfully trained the model on GPU, exporting `photon_model.pt` (1.3 MB) and `conformal_quantiles.json`.

### Milestone 4: SpectraQuest Delimiter & Parsing Bug
- **User Issue:** *"Error parsing file: could not convert string '2.35039e+000\t1.45487e+000\t-1.66708e+000\t-2.05561e+000\t' to float64 at row 0, column 1."*
- **Root Cause:** Real accelerometer files from SpectraQuest test rigs (`testData/gearboxdata/b30hz0.txt`) use scientific notation, mixed tab delimiters, and trailing tab characters (`\t\r\n`), causing standard `np.loadtxt` and naive `pd.read_csv` calls to fail.
- **Resolution:** Implemented `robust_parse_vibration_data` utilizing regular-expression whitespace parsing (`sep=r"\s+"`), automatic column dropping for NaN-padding, header detection, and fallback line-by-line float conversion.

### Milestone 5: Dynamic UI vs. Static Visuals
- **User Issue:** *"Remove the visualls they are static. The visualles that you added form the collab notebook is static remove it."*
- **Resolution:** Completely purged pre-rendered static PNG/JPEG references. Replaced with real-time dynamic signal processing: any uploaded `.csv` or `.txt` file is streamed directly through numpy/scipy FFT engines to generate live, interactive Plotly waveforms and order spectra.

### Milestone 6: GearGuard AI Industrial Dashboard & Git Repository Isolation
- **User Request:** *"Ok look in the home/jeremy you will find a folder called GearGuardAI. Use the ui from that and desing the back end accordingly. Even if we enter data of any channel it should adapt to the data and provide the output and analysis for us. DON'T COMIT IT TO THE CURRENT REPO IN GIT HUM. CREATE A NEW REPO CALLED GearGuardAI AND COMMIT IT TO THIS REPO."*
- **Engineering Actions:**
  1. Initialized independent Git repository at `/home/jeremy/GearGuardAI`.
  2. Linked to remote `https://github.com/JerrySeba/GearGuardAI.git` on branch `main`.
  3. Added `.gitignore` to prevent committing raw 109 MB dataset folders (`BrokenTooth Data/`, `Healthy Data/`).
  4. Preserved dark cyber-industrial SCADA aesthetic (Orbitron fonts, neon telemetry gauges, 5 navigation pages).
  5. Built and trained `gearguard_real_model.joblib` across multiple load states (0%, 50%, 90% load).
  6. Implemented `adapt_channel_projection` to support 1-channel, 2-channel, 3-channel, 4-channel, and 8-channel files without crashing.
  7. Populated `samples/` with verified test benchmarks (`healthy_4ch_sample.txt`, `broken_4ch_sample.txt`, `chipped_1ch_sample.csv`).

### Milestone 7: Production Port 8501 Resolution
- **User Issue:** *"There is not change in the UI"*
- **Root Cause:** A background Streamlit process (PID 1284196) was already running the older `GCEM_Hacks5.0` app on default port `8501`. When `GearGuardAI` was launched on port `8502`, the user refreshed `localhost:8501` and saw the old UI.
- **Resolution:** Terminated all stale Streamlit processes (`kill -9 1284196 2194206`), and launched `GearGuardAI/app.py` directly on default port `8501`. Verified HTTP 200 response via `curl -I http://localhost:8501`.

---

## 5. Repository Structure & Verification

### 5.1 Repository 1: `GCEM_Hacks5.0`
- **GitHub URL:** `https://github.com/JerrySeba/GCEM_Hacks5.0.git`
- **Location:** `/home/jeremy/GCEM_Hacks5.0`
- **Core Files:**
  - `train_pipeline.py`: PyTorch training pipeline with COT, 8-DOF Physics Loss, SE-ResNet-18, and Conformal Prediction.
  - `photon_model.pt`: Trained PyTorch model checkpoint (1.3 MB).
  - `conformal_quantiles.json`: Conformal calibration quantiles ($q_{sev} = 0.3473$, $q_{rul} = 2096$).
  - `GCEM_Hacks5.0_Train.ipynb`: Google Colab GPU training notebook.
  - `ROUND_2_PRESENTATION_DECK.md`: 4-slide defense deck for jury review.
  - `GEARBOX_FAULT_DIAGNOSIS_STRATEGY.md`: 52 KB in-depth scientific treatise.
  - `app.py`: Research-oriented Streamlit dashboard.

### 5.2 Repository 2: `GearGuardAI`
- **GitHub URL:** `https://github.com/JerrySeba/GearGuardAI.git`
- **Location:** `/home/jeremy/GearGuardAI`
- **Core Files:**
  - `app.py`: Cyber-industrial SCADA application with universal multi-channel ingestion ($1 \le C \le 8+$) and real-time Plotly signal analytics.
  - `gearguard_real_model.joblib`: Multi-domain 28-feature Balanced Random Forest trained across 0% to 90% load states.
  - `train_real_model.py`: Training script for feature extraction and load-generalized model fitting.
  - `test_variable_load.py`: Automated test suite verifying classification accuracy across load conditions.
  - `samples/`: Verified benchmark files for one-click testing:
    - `samples/healthy_4ch_sample.txt` (4-channel baseline healthy gearbox)
    - `samples/broken_4ch_sample.txt` (4-channel broken tooth gearbox)
    - `samples/chipped_1ch_sample.csv` (1-channel single accelerometer test)

---

## 6. How to Run and Evaluate the Systems

### 6.1 Running the Live GearGuard AI SCADA Dashboard (Port 8501)
To view and interact with the primary industrial dashboard:
```bash
cd /home/jeremy/GearGuardAI
/home/jeremy/GCEM_Hacks5.0/.venv/bin/streamlit run app.py --server.headless true --server.port 8501
```
Open browser at: `http://localhost:8501`

#### Test Walkthrough:
1. Navigate to **Diagnosis Log** or **Dashboard** in the sidebar.
2. Select **"Upload Vibration Data"** or click **"Load Benchmark Sample"**.
3. Choose `samples/broken_4ch_sample.txt` or `samples/healthy_4ch_sample.txt`.
4. The system automatically detects file structure, cleans delimiters, adapts channel projection, computes 28 statistical features, runs FFT spectra, and displays:
   - Predicted Fault Class (e.g., *Broken Tooth Fault* vs. *Healthy Baseline*)
   - Damage Severity Index & ISO 10816-3 Risk Gauge
   - Remaining Useful Life (RUL) Cycles
   - Conformal Prediction Confidence Interval
   - Interactive Zoomable Plotly Time-Domain and Frequency-Domain Charts

### 6.2 Running the PhOTON-C²-UQ Research Pipeline
To verify or retrain the deep learning physics-informed model:
```bash
cd /home/jeremy/GCEM_Hacks5.0
/home/jeremy/GCEM_Hacks5.0/.venv/bin/python3 train_pipeline.py --epochs 10 --batch-size 32
```

---

## 7. Jury Defense & Evaluation Cheat Sheet

| Question from Evaluators | Technical Answer | Where to Find Code |
| :--- | :--- | :--- |
| **"How do you handle variable shaft speed?"** | Computed Order Tracking (COT). We interpolate time-domain signals onto constant shaft angle increments $\Delta \theta$ via cubic spline resampling. Mesh frequencies $f_m = Z f_r$ become stationary order harmonics $O_m = Z$. | `GCEM_Hacks5.0/train_pipeline.py:L142-L198` |
| **"How does the model handle variable load?"** | Load affects vibration amplitude, not frequency. We normalize signals via energy scaling and train on multi-load datasets (0%, 50%, 90% load) with multi-domain feature invariance. | `GearGuardAI/train_real_model.py:L45-L98` |
| **"How do you solve scarce / limited failure data?"** | We use an 8-DOF Lumped-Parameter Dynamic Digital Twin solving differential equations under varying tooth crack/spalling levels via RK45 to synthesize physically realistic failure modes. | `GCEM_Hacks5.0/train_pipeline.py:L210-L290` |
| **"Why is your Uncertainty Quantification better than Softmax?"** | Softmax produces overconfident heuristic probabilities. We use Split-Conformal Prediction, providing distribution-free finite-sample coverage guarantees with calibrated non-conformity quantiles. | `GCEM_Hacks5.0/conformal_quantiles.json` |
| **"What happens if an operator uploads a 1-channel or 8-channel file?"** | Our Universal Channel Projection engine automatically adapts the tensor dimension without crashing, replicating 1-channel data or ranking 8-channel data by signal variance. | `GearGuardAI/app.py:adapt_channel_projection` |

---

*Report prepared for GCEM Hacks 5.0 Evaluation Team.*
