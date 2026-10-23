"""DriftLens dashboard. Run with:  streamlit run app.py

Everything shown here comes from the real simulator/detectors/alert engine.
Experiment tables are read from results/*.csv (created by experiments/run_experiments.py).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.alert_engine import batch_drift_signal, new_alert_state, update_alert_state
from src.data import load_dataset, make_reference_and_batches
from src.drift_detectors import detect_drift
from src.model import evaluate_model, train_model

RESULTS_DIR = Path(__file__).parent / "results"

st.set_page_config(page_title="DriftLens", layout="wide")
st.title("DriftLens: ML Data Drift Monitoring with Persistent Alerts")
st.caption(
    "Simulated batch stream. KS and PSI detect INPUT-distribution changes; they do not "
    "prove concept drift or accuracy loss. Labels are used only for offline model metrics."
)


@st.cache_resource
def get_data_and_model():
    X_train, X_test, y_train, y_test = load_dataset()
    model = train_model(X_train, y_train)
    baseline = evaluate_model(model, X_test, y_test)
    return X_train, X_test, y_test, model, baseline


X_train, X_test, y_test, model, baseline = get_data_and_model()

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Stream settings")
    scenario = st.selectbox("Scenario", ["stable", "gradual", "sudden", "temporary"])
    shift_strength = st.slider("Shift strength (x feature std)", 0.0, 3.0, 1.0, 0.1)
    n_batches = st.slider("Number of batches", 10, 40, 20)
    batch_size = st.slider("Batch size", 30, 300, 100, 10)
    drift_start = st.slider("Drift start batch", 1, 20, 8)
    drift_length = st.slider("Temporary drift length", 1, 10, 4)
    seed = st.number_input("Random seed", 0, 100000, 0)

    st.header("Detection / alert settings")
    alpha = st.number_input("KS significance level (after correction)", 0.001, 0.2, 0.05, 0.005)
    psi_threshold = st.number_input("PSI threshold", 0.05, 1.0, 0.25, 0.05)
    min_features = st.slider("Min flagged features for a batch signal", 1, 10, 1)
    persistence_k = st.slider("Persistence: consecutive batches", 2, 5, 2)

    reset = st.button("Reset stream with these settings", type="primary")


def reset_stream():
    ref, batches = make_reference_and_batches(
        X_train, X_test, y_test,
        scenario=scenario, n_batches=n_batches, batch_size=batch_size,
        shift_strength=shift_strength, seed=int(seed),
        drift_start=drift_start, drift_length=drift_length,
    )
    st.session_state.reference = ref
    st.session_state.batches = batches
    st.session_state.pos = 0
    st.session_state.alert_state = new_alert_state()
    st.session_state.history = []
    st.session_state.feature_tables = {}
    # Settings are frozen at reset so thresholds don't change mid-stream.
    st.session_state.cfg = dict(
        scenario=scenario, alpha=alpha, psi_threshold=psi_threshold,
        min_features=min_features, persistence_k=persistence_k, n_batches=n_batches,
    )


if reset or "batches" not in st.session_state:
    reset_stream()


def process_next_batch():
    ss = st.session_state
    cfg = ss.cfg
    b = ss.batches[ss.pos]
    fr = detect_drift(ss.reference, b["X"], alpha=cfg["alpha"], psi_threshold=cfg["psi_threshold"])
    signal = batch_drift_signal(fr, cfg["min_features"])
    ss.alert_state = update_alert_state(signal, ss.alert_state, cfg["persistence_k"])
    flagged = fr.loc[fr["combined_feature_flag"], "feature"].tolist()
    m = evaluate_model(model, b["X"], b["y"])
    ss.feature_tables[b["batch_id"]] = fr
    ss.history.append(
        {
            "batch_id": b["batch_id"],
            "is_shifted": bool(b["is_shifted"]),
            "signal": signal,
            "immediate_alert": ss.alert_state["immediate_alert"],
            "persistent_alert": ss.alert_state["persistent_alert"],
            "consecutive_count": ss.alert_state["consecutive_count"],
            "n_flagged": len(flagged),
            "flagged_features": ", ".join(flagged),
            "shifted_features": ", ".join(b["shifted_features"]),
            "accuracy": m["accuracy"],
            "f1": m["f1"],
        }
    )
    ss.pos += 1


tab_live, tab_exp = st.tabs(["Live stream", "Experiment results"])

# ------------------------------------------------------------- live tab
with tab_live:
    ss = st.session_state
    total = len(ss.batches)
    c1, c2, c3 = st.columns([1, 1, 3])
    if c1.button("Next batch", disabled=ss.pos >= total):
        process_next_batch()
    if c2.button("Run to end", disabled=ss.pos >= total):
        while ss.pos < total:
            process_next_batch()
    c3.progress(ss.pos / total, text=f"Batch {ss.pos} of {total} processed "
                                      f"(scenario: {ss.cfg['scenario']})")

    if not ss.history:
        st.info("Press **Next batch** to start the stream.")
    else:
        hist = pd.DataFrame(ss.history)
        last = hist.iloc[-1]
        bid = int(last["batch_id"])

        a1, a2, a3 = st.columns(3)
        a1.metric("Immediate alert", "ALERT" if last["immediate_alert"] else "OK")
        a2.metric(
            f"Persistent alert (k={ss.cfg['persistence_k']})",
            "ALERT" if last["persistent_alert"] else "OK",
        )
        a3.metric("Consecutive drift batches", int(last["consecutive_count"]))

        # Allow browsing earlier batches
        shown_id = st.select_slider("Inspect batch", options=list(hist["batch_id"]), value=bid)
        fr = ss.feature_tables[shown_id]
        batch = next(b for b in ss.batches if b["batch_id"] == shown_id)

        left, right = st.columns(2)
        with left:
            st.subheader("Feature-level KS and PSI")
            st.dataframe(
                fr.sort_values("psi", ascending=False).round(4),
                use_container_width=True, height=360,
            )
        with right:
            st.subheader("PSI by feature")
            top = fr.sort_values("psi", ascending=False).head(15)
            fig = px.bar(
                top, x="psi", y="feature", orientation="h",
                color="combined_feature_flag",
                color_discrete_map={True: "#d62728", False: "#7f7f7f"},
            )
            fig.add_vline(x=ss.cfg["psi_threshold"], line_dash="dash",
                          annotation_text="PSI threshold")
            fig.update_layout(yaxis=dict(autorange="reversed"), height=360,
                              legend_title="flagged")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Reference vs current distribution")
        feat = st.selectbox("Feature", list(fr.sort_values("psi", ascending=False)["feature"]))
        fig = go.Figure()
        fig.add_histogram(x=ss.reference[feat], name="Reference (train)", opacity=0.55,
                          histnorm="probability density")
        fig.add_histogram(x=batch["X"][feat], name=f"Batch {shown_id}", opacity=0.55,
                          histnorm="probability density")
        fig.update_layout(barmode="overlay", height=320)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Alert timeline")
        rows = [("Injected shift (truth)", "is_shifted"),
                ("Immediate alert", "immediate_alert"),
                ("Persistent alert", "persistent_alert")]
        fig = go.Figure()
        for label, col in rows:
            d = hist[hist[col]]
            fig.add_trace(go.Scatter(x=d["batch_id"], y=[label] * len(d),
                                     mode="markers", name=label, marker=dict(size=14)))
        fig.update_yaxes(categoryorder="array", categoryarray=[r[0] for r in rows][::-1])
        fig.update_xaxes(range=[-0.5, total - 0.5], dtick=1, title="Batch")
        fig.update_layout(height=260)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Model performance on labelled synthetic batches (offline evaluation)")
        st.caption(
            f"Baseline on held-out test set: accuracy {baseline['accuracy']:.3f}, "
            f"F1 {baseline['f1']:.3f}. A distribution shift does not necessarily reduce accuracy."
        )
        fig = px.line(hist, x="batch_id", y=["accuracy", "f1"], markers=True)
        fig.add_hline(y=baseline["f1"], line_dash="dot", annotation_text="baseline F1")
        fig.update_layout(height=280, yaxis_range=[0, 1.05])
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Alert history")
        st.dataframe(hist.round(3), use_container_width=True)

# --------------------------------------------------- experiment results tab
with tab_exp:
    summary_path = RESULTS_DIR / "summary.csv"
    if not summary_path.exists():
        st.warning("No experiment results yet. Run `python experiments/run_experiments.py` "
                   "from the project root, then refresh this page.")
    else:
        summary = pd.read_csv(summary_path)
        st.subheader("Immediate vs persistence (measured)")
        st.dataframe(summary.round(3), use_container_width=True)
        st.caption(
            "false_alert_batch_rate: alerts on non-shifted batches / non-shifted batches. "
            "false_alert_run_rate: runs with at least one false alert. "
            "detection_rate: runs alerted within the evaluation window after drift start. "
            "mean_detection_delay: batches from drift start to first alert, detected runs only."
        )
        for name, title in [("feature_id.csv", "Feature identification"),
                            ("performance.csv", "Model performance, shifted vs not shifted")]:
            path = RESULTS_DIR / name
            if path.exists():
                st.subheader(title)
                st.dataframe(pd.read_csv(path).round(3), use_container_width=True)
        for name in ["summary.csv", "batch_results.csv"]:
            path = RESULTS_DIR / name
            if path.exists():
                st.download_button(f"Download {name}", path.read_bytes(), file_name=name,
                                   mime="text/csv", key=name)
