# DriftLens: Teacher Review & Live Demonstration Guide

This guide gives you a **step-by-step script** for your teacher review. It explains exactly **what to click**, **what to say**, **how drift occurs**, **how the dashboard highlights shifted/wrong values**, and **how to answer tough questions confidently**.

---

## 1. Quick Setup Before Your Teacher Arrives

1. Open your terminal in the project folder.
2. Activate your virtual environment and start the app:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   streamlit run app.py
   ```
3. Open the browser at `http://localhost:8501`.
4. Maximize your browser window so all charts and metrics are clearly visible.

---

## 2. The 30-Second Opening Pitch (Say this first!)

> *"Good morning / afternoon Professor,*
>
> *Today I am presenting **DriftLens**, an online machine learning data drift monitoring system with persistent alerts.*
>
> *In production ML, models are trained on past data, but real-world data changes over time. When input data changes, we call this **Data Drift**.*
>
> *Most existing monitoring tools have a big flaw: **they trigger an alert on the very first batch that looks slightly abnormal**. In practice, 25% to 30% of those alerts are false alarms caused by random noise, leading to 'alert fatigue'.*
>
> *Our research project asks: **Can requiring drift evidence to persist across 2 consecutive batches eliminate false alarms without causing dangerous delays?***
>
> *Let me demonstrate how the dashboard works and what happens when data drifts."*

---

## 3. Step-by-Step Live Demo Script

---

### Step 1: Explain the Baseline Data (First 30 seconds)

**What to point at on the screen:**
- Point at the top caption and the sidebar.

**What to say:**
> *"We use the standard Breast Cancer Wisconsin dataset with 30 numerical medical features like cell radius, texture, and perimeter.*
>
> *Our baseline Random Forest classifier was trained on 80% of the data and achieves **95.6% accuracy** and a **96.6% F1-score** on the held-out test data. This training set is our **Reference Baseline**."*

---

### Step 2: Demo Scenario 1 — Normal / Stable Data (Showing False Alarms)

**What to click:**
1. In the sidebar:
   - Set **Scenario** to `stable`.
   - Leave **Random seed** at `0`.
   - Click **Reset stream with these settings** (red button).
2. Click **Run to end** or click **Next batch** a few times.

**What to show your teacher on the screen:**
- Look at the **Alert timeline** chart at the bottom.
- Look at the **Immediate alert** vs. **Persistent alert** metrics.

**What to say:**
> *"Here, the data is completely normal—no drift has been injected.*
>
> *Notice what happens with the **Immediate Alert strategy**: even though the data is clean, random sampling noise causes the immediate detector to occasionally flash **ALERT**. In our 20-seed experiments, the immediate strategy generated false alarms on **27.8% of clean batches**!*
>
> *Now look at our **Persistent Strategy ($k=2$)**: because the noise was just a one-batch fluke, the next batch went back to normal, the counter reset to zero, and the persistent strategy stayed calm (**OK**).*
>
> *This immediately proves our first hypothesis: persistence eliminates single-batch noise."*

---

### Step 3: Demo Scenario 2 — Sudden Drift (How Drift Occurs & How Values Look "Wrong")

This is the most important part of your review.

**What to click:**
1. In the sidebar:
   - Set **Scenario** to `sudden`.
   - Set **Shift strength** to `1.5` or `1.0`.
   - Set **Drift start batch** to `8`.
   - Click **Reset stream with these settings**.
2. Click **Next batch** 7 times until you reach **Batch 7** (no drift yet).
   - Point out that Consecutive Drift Batches is `0`.
3. Click **Next batch** once more to advance to **Batch 8** (the drift begins!).

#### How Drift Occurs (Explain the Math Simply):
**What to say:**
> *"Professor, at Batch 8, sudden drift has been injected. How does drift occur?*
>
> *In the real world, this happens when a medical imaging sensor gets recalibrated, or a new clinic sends images with higher contrast.*
>
> *In our simulator, we take the top features (`mean radius`, `mean texture`, `mean perimeter`) and add a shift equal to **1.5 standard deviations** of the baseline. That means cell sizes suddenly appear significantly larger than anything the model saw during training."*

#### Showing the "Wrong / Shifted Values" on the Dashboard:
Now show the teacher the 3 visual proofs on the dashboard:

1. **Proof 1 — The Reference vs Current Histogram:**
   - Under *Reference vs current distribution*, select `mean radius`.
   - Point at the blue vs. orange bell curves.
   > *"Look at this distribution plot. The blue curve is the normal training baseline. The orange curve is the current batch. You can visually see that the entire distribution has shifted to the right into 'abnormal' territory."*

2. **Proof 2 — Feature-Level KS and PSI Table & Bar Chart:**
   - Point at the red bars in the **PSI by feature** chart.
   > *"Our dual detector catches this immediately:*
   > *1. The **Kolmogorov-Smirnov test** checks the curve shape. After Benjamini-Hochberg multiple-testing correction, the p-value drops to 0.000 (statistically significant).*
   > *2. The **Population Stability Index (PSI)** measures the divergence. Normal is below 0.10. Here, PSI jumps to over 1.5, shown by the red bar crossing the dashed threshold line.*
   > *Because both tests agree, the feature gets a **Combined Feature Flag**."*

3. **Proof 3 — The Alert Comparison at Batch 8:**
   - Point at the top metric cards at Batch 8:
   > *"Look at the alert state at Batch 8:*
   > - *Immediate Alert is already screaming **ALERT**.*
   > - *Persistent Alert is still showing **OK**! Why? Because this is only the first drifted batch (`consecutive_count = 1`). It is waiting to verify whether this is real drift or just noise."*

4. **Now click "Next batch" to move to Batch 9:**
   - Look at the metric cards update!
   > *"Now at Batch 9, the drift is still present. The consecutive counter reaches `2`.*
   > *Now the **Persistent Alert switches to ALERT**!*
   > *It successfully confirmed genuine drift with a delay of only **1 batch**."*

---

### Step 4: Demo Scenario 3 — Temporary Shift (Showing Counter Resets)

**What to click:**
1. In the sidebar:
   - Set **Scenario** to `temporary`.
   - Set **Drift start batch** to `5`.
   - Set **Temporary drift length** to `3` (batches 5, 6, and 7 will be shifted; batch 8 returns to normal).
   - Click **Reset stream with these settings**.
2. Click **Run to end**.

**What to show your teacher on the screen:**
- Scroll down to the **Alert timeline** chart.
- Show batches 5, 6, 7 (shifted), and then batch 8 (normal).

**What to say:**
> *"This scenario tests what happens during a temporary anomaly—like a brief power fluctuation.*
>
> *Notice that during batches 5, 6, and 7, the drift is active. But the moment batch 8 returns to normal baseline, our alert state machine **immediately resets the consecutive counter back to 0**.*
>
> *This proves our system does not get 'stuck' in an alarm state once conditions return to normal."*

---

### Step 5: Explain Model Performance vs. Input Drift (Crucial Concept!)

Point to the graph titled: **"Model performance on labelled synthetic batches (offline evaluation)"**.

**What to say:**
> *"A critical research distinction in DriftLens is the difference between **Data Drift** and **Accuracy Drop**:*
> - *Detecting drift tells us that the **input features changed** ($P(X)$).*
> - *It does **not** automatically mean the model's accuracy immediately collapsed! In medical data, if tumor cell radius shifts higher, a classifier trained to flag large cells as malignant might actually still predict correctly.*
> - *That is why **labels are never given to our drift detector**. Drift monitoring runs completely unsupervised in real time. We only use labels offline to measure whether the shift hurt accuracy."*

---

### Step 6: The "Experiment Results" Tab (The Final Punchline)

**What to click:**
- Click on the second tab: **"Experiment results"**.

**What to show your teacher:**
- The summary table comparing Immediate vs. Persistent.

**What to say:**
> *"Finally, Professor, we didn't just test this on one lucky run. We built an automated experiment runner that tested **20 random seeds across 1,600 batches**.*
>
> *Here are our definitive findings:*
> 1. *On clean data, the Immediate strategy produced a **27.8% false alarm rate**. The Persistent strategy dropped that to **8.5%**—a **70% reduction in false alarms**.*
> 2. *On sudden drift, the Persistent strategy detected 100% of drift events with an average detection delay of only **0.9 batches**.*
> 3. *Our feature identification achieved **100% recall** and **81.7% precision** in pinpointing exactly which medical features were altered.*
>
> *In conclusion, requiring drift evidence to persist across just 2 consecutive batches successfully solves the alert fatigue problem without compromising system safety."*

---

## 4. Top 5 Questions Your Teacher Might Ask & How to Answer

### Q1: "Why did you use both the KS-Test and PSI? Isn't one enough?"
> **Answer:**
> *"They complement each other's weaknesses, Professor:*
> - *The **KS-test** is a formal statistical hypothesis test that checks whether two distributions differ, but with large sample sizes it can flag tiny, clinically meaningless differences ($p < 0.05$).*
> - *The **PSI** measures the actual magnitude/practical distance between distributions across quantile bins, but doesn't provide a formal p-value.*
> - *By requiring **both** (significant adjusted KS p-value AND PSI $\ge 0.10$ or $0.25$), we avoid flagging statistically significant but practically tiny shifts."*

### Q2: "What is Benjamini-Hochberg correction and why is it necessary?"
> **Answer:**
> *"Because the Breast Cancer dataset has 30 features, we are running 30 simultaneous hypothesis tests on every single batch. Under a standard $\alpha = 0.05$, there is a high probability that 1 or 2 features will look significant purely by random chance (the multiple comparisons problem). Benjamini-Hochberg False Discovery Rate (FDR) correction adjusts the p-values so we don't get false positive feature flags."*

### Q3: "What happens if real drift happens, but your persistent strategy waits for 2 batches? Isn't that dangerous?"
> **Answer:**
> *"That is the fundamental engineering trade-off: **speed vs. precision**. In our experiments, persistent alerting added a delay of only 0.9 batches (less than 1 batch). In exchange, it eliminated 70% of false alarms. If a team receives 30 false alarms a day, they end up disabling the monitoring system entirely, which is far more dangerous."*

### Q4: "Why don't you update the baseline reference after every batch?"
> **Answer:**
> *"If we updated the baseline after every batch, a **gradual drift** would slowly pollute the baseline. The detector would adapt to the drift and treat the corrupted data as the new normal. We keep the baseline anchored to the validated training set."*

### Q5: "How did you test your code?"
> **Answer:**
> *"We wrote an automated test suite with **27 pytest unit and integration tests** covering the alert state machine, consecutive counter resets, PSI edge cases (like empty bins), KS corrections, reproducible batch generation, and baseline Random Forest evaluation. All 27 tests pass in 3.4 seconds."*

---

## 5. Cheat Sheet of Numbers to Memorize

| Metric | Immediate Strategy | Persistent Strategy ($k=2$) |
| :--- | :---: | :---: |
| **False Alarm Rate (Clean Data)** | **27.8%** | **8.5%** (~70% reduction!) |
| **Detection Rate (Sudden Drift)** | 100% | 100% |
| **Detection Delay (Sudden Drift)** | 0.0 batches | **0.9 batches** (< 1 batch delay) |
| **Feature Identification Recall** | 100% | 100% |
| **Feature Identification Precision** | 81.7% | 81.7% |
| **Baseline Model Accuracy** | 95.6% | 95.6% |
| **Baseline Model F1-Score** | 96.6% | 96.6% |
