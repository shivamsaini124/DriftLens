# DriftLens: Online-Style ML Data Drift Monitoring with Persistent Alerts

DriftLens is an online-style machine learning data drift monitoring system and interactive Streamlit application. It simulates incoming feature batches, detects distribution shifts using Kolmogorov–Smirnov (KS) tests (with Benjamini–Hochberg FDR multiple-testing correction) and Population Stability Index (PSI), and compares an **immediate alert** policy against a **persistence-based alert strategy** (requiring consecutive qualifying batches).

---

## 1. Problem Statement and Motivation

In production machine learning systems, data distributions naturally evolve over time. While single-batch drift detectors can catch distribution shifts rapidly, single batches frequently trigger false alarms due to transient sampling noise or short-lived anomalies. Alert fatigue often leads engineering teams to ignore monitoring alerts altogether.

DriftLens investigates whether requiring statistical drift evidence to **persist across consecutive batches** can suppress spurious false alarms without causing an unacceptable detection delay when genuine drift occurs.

> **Scope Note:** Drift monitoring assesses *input distribution changes* ($P(X)$). Input drift does not inherently prove concept drift ($P(Y|X)$) or declining model accuracy. Model labels are strictly withheld from detectors and used solely for offline model performance benchmarking.

---

## 2. Research Question

> **Can requiring drift evidence to persist across consecutive batches reduce false alarms compared with single-batch alerting, without causing an unacceptable increase in detection delay?**

---

## 3. Dataset and Experimental Setup

- **Dataset:** Scikit-learn Breast Cancer Wisconsin Diagnostic dataset (569 samples, 30 continuous features, binary target).
- **Baseline Classifier:** Random Forest (`n_estimators=100`, stratified train/test split 80/20, `random_state=42`).
  - Baseline Test Accuracy: **95.6%**
  - Baseline Test F1-Score: **96.6%**
- **Simulation Scenarios:**
  - **Stable:** Incoming batches drawn from the unshifted test distribution.
  - **Gradual:** Feature distribution shifts linearly over successive batches starting at `shift_start`.
  - **Sudden:** Abrupt shift introduced starting at `shift_start` and sustained to the end.
  - **Temporary:** Abrupt shift active for `shift_duration` batches, followed by a return to baseline.
- **Controlled Drift Injection:** An offset proportional to the reference feature's standard deviation ($\Delta = \mu_{\text{shift}} \cdot \sigma_{\text{ref}}$) is added to target features.

---

## 4. Drift Detection Methodology

DriftLens uses a dual-detector methodology:

1. **Kolmogorov–Smirnov (KS) Test with FDR Correction:**
   - Evaluates whether the empirical cumulative distribution functions of the reference and current batch differ significantly (`scipy.stats.ks_2samp`).
   - Controls false discovery rates across all 30 simultaneous feature tests using the **Benjamini–Hochberg (BH)** procedure (`alpha = 0.05`).
2. **Population Stability Index (PSI):**
   - Quantile bin edges are fixed from the reference training distribution.
   - Smooths zero-count bins ($\epsilon = 10^{-6}$) and measures distribution divergence:
     $$\text{PSI} = \sum_{i=1}^B (P_i - Q_i) \ln\left(\frac{P_i}{Q_i}\right)$$
   - Features exceeding the configured threshold (default $\text{PSI} \ge 0.25$ or $0.10$) trigger a PSI flag.
3. **Combined Feature Flag:**
   - A feature is flagged if and only if **both** the adjusted KS $p$-value is statistically significant AND PSI reaches the threshold:
     $$\text{Flag}_f = (\text{KS\_adj\_pvalue}_f < \alpha) \land (\text{PSI}_f \ge \tau)$$

---

## 5. Alert Strategies

Batch signals are collapsed from feature-level flags:
$$\text{BatchSignal} = \left(\sum_f \text{Flag}_f \ge \text{min\_features}\right)$$

- **Immediate Strategy:** Alerts immediately on any batch where $\text{BatchSignal} = \text{True}$.
- **Persistent Strategy ($k=2$):** Alerts only when $\text{BatchSignal} = \text{True}$ for at least $k$ consecutive batches. If a batch exhibits no drift, the consecutive counter immediately resets to zero.

---

## 6. Experimental Results

Experiments evaluated 20 independent seeds across all 4 scenarios (20 batches per run, batch size 100, shift strength 1.0, drift starting at batch 8):

### Summary Metrics (`results/summary.csv`)

| Strategy | Scenario | False Alert Batch Rate | False Alert Run Rate | Detection Rate | Mean Detection Delay (Batches) | Missed Runs |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Immediate** | Stable | **27.8%** | **100%** | N/A | N/A | 0 |
| **Immediate** | Gradual | 24.3% | 85.0% | 100% | 1.30 | 0 |
| **Immediate** | Sudden | 24.3% | 85.0% | 100% | 0.00 | 0 |
| **Immediate** | Temporary | 27.2% | 100% | 100% | 0.00 | 0 |
| **Persistent ($k=2$)** | Stable | **8.5%** | **80.0%** | N/A | N/A | 0 |
| **Persistent ($k=2$)** | Gradual | **7.1%** | **40.0%** | **90.0%** | 2.33 | 2 |
| **Persistent ($k=2$)** | Sudden | **7.1%** | **40.0%** | **100%** | 0.90 | 0 |
| **Persistent ($k=2$)** | Temporary | **10.9%** | **85.0%** | **100%** | 0.90 | 0 |

### Key Findings
1. **False Alarm Reduction:** Requiring $k=2$ consecutive qualifying batches reduced the false alert batch rate from **27.8% down to 8.5%** on stable data (a **70% reduction** in false alerts).
2. **Detection Delay Trade-off:** On sudden shift, the persistent alert requires just **0.9 additional batches** (delay from 0.00 to 0.90) while slashing false alarms by over 70%.
3. **Feature Identification (`results/feature_id.csv`):**
   - Sudden Shift: **100% Recall**, **81.7% Precision**
   - Temporary Shift: **100% Recall**, **81.2% Precision**
   - Gradual Shift: **81.7% Recall**, **77.9% Precision**

---

## 7. Trade-offs and Limitations

- **Detection Delay:** Persistent alerting introduces an unavoidable minimum delay of $k-1$ batches before triggering. For mission-critical tasks requiring sub-batch intervention, immediate alerting may be preferable despite noise.
- **Synthetic vs. Real-world Drift:** Experiments use synthetic mean-shifts. Real-world drift may involve complex covariance changes, multimodal shifts, or categorical variables.
- **Reference Window Aging:** Baselines are fixed from the training set. Production deployments must determine when to re-baseline without incorporating gradual drift into the normal profile.

---

## 8. Installation and Run Instructions

### 1. Prerequisites
- Python 3.11 or 3.12
- Git

### 2. Setup Environment

#### Windows (PowerShell)
```powershell
git clone https://github.com/YOUR_USERNAME/DriftLens.git
cd DriftLens

py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

#### macOS / Linux
```bash
git clone https://github.com/YOUR_USERNAME/DriftLens.git
cd DriftLens

python3.12 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Run Test Suite
```bash
pytest
```
All 27 unit and integration tests will execute across data simulation, model evaluation, KS/PSI drift detectors, and alert state management.

### 4. Run Reproducible Experiments
```bash
python experiments/run_experiments.py
```
This generates:
- `results/summary.csv`
- `results/batch_results.csv`
- `results/feature_id.csv`
- `results/performance.csv`

### 5. Launch the Streamlit Dashboard
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser. Features:
- **Interactive Stream Simulator:** Step batch-by-batch or run to completion with stable, gradual, sudden, or temporary shifts.
- **Live Visualizations:** PSI ranking bar charts, reference vs. current histogram overlays, and alert timeline scatter plots.
- **Comparison Tab:** View and download measured experimental CSV results.
