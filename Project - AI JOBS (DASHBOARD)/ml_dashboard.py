"""
Interactive ML Dashboard — Global AI Jobs
==========================================
Streamlit app with EDA, model performance, feature importance,
and live salary prediction.

Usage:
    streamlit run ml_dashboard.py
"""
import os
import json

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Jobs ML Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "global_ai_jobs.csv")
ARTIFACT_DIR = os.path.join(BASE_DIR, "ml_artifacts")


# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Dark-first professional theme */
    .main .block-container { padding: 1.5rem 2rem; max-width: 1400px; }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .metric-card h3 { color: #e94560; margin: 0 0 0.3rem 0; font-size: 0.85rem; }
    .metric-card p { color: #eee; margin: 0; font-size: 1.6rem; font-weight: 700; }
    .hero-header {
        background: linear-gradient(135deg, #0f3460, #533483, #e94560);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
    }
    .sub-text { color: #888; font-size: 0.95rem; margin-bottom: 1.5rem; }
    div[data-testid="stTabs"] button {
        font-weight: 600 !important;
        font-size: 1rem !important;
    }
</style>
""", unsafe_allow_html=True)


# ── Data Loading (cached) ───────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    if "id" in df.columns:
        df.drop(columns=["id"], inplace=True)
    return df


@st.cache_resource
def load_artifacts():
    artifacts = {}
    try:
        artifacts["model"] = joblib.load(os.path.join(ARTIFACT_DIR, "best_model.pkl"))
        artifacts["scaler"] = joblib.load(os.path.join(ARTIFACT_DIR, "scaler.pkl"))
        artifacts["label_encoders"] = joblib.load(os.path.join(ARTIFACT_DIR, "label_encoders.pkl"))
        with open(os.path.join(ARTIFACT_DIR, "metrics.json")) as f:
            artifacts["metrics"] = json.load(f)
        with open(os.path.join(ARTIFACT_DIR, "columns.json")) as f:
            artifacts["columns"] = json.load(f)
        artifacts["feature_importance"] = pd.read_csv(
            os.path.join(ARTIFACT_DIR, "feature_importance.csv")
        )
        artifacts["loaded"] = True
    except Exception as e:
        artifacts["loaded"] = False
        artifacts["error"] = str(e)
    return artifacts


# ── Helper ───────────────────────────────────────────────────────────────────
def metric_card(label, value):
    st.markdown(
        f'<div class="metric-card"><h3>{label}</h3><p>{value}</p></div>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="hero-header">🤖 Global AI Jobs — ML Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Salary prediction & analytics powered by machine learning</div>', unsafe_allow_html=True)

df = load_data()
artifacts = load_artifacts()

# ── Sidebar Filters ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🔎 Filters")
    countries = st.multiselect("Country", sorted(df["country"].unique()), default=[])
    roles = st.multiselect("Job Role", sorted(df["job_role"].unique()), default=[])
    exp_levels = st.multiselect("Experience Level", sorted(df["experience_level"].unique()), default=[])

    # Apply filters
    filtered = df.copy()
    if countries:
        filtered = filtered[filtered["country"].isin(countries)]
    if roles:
        filtered = filtered[filtered["job_role"].isin(roles)]
    if exp_levels:
        filtered = filtered[filtered["experience_level"].isin(exp_levels)]

    st.divider()
    st.caption(f"Showing **{len(filtered):,}** of {len(df):,} records")

# ═══════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════
tab_eda, tab_perf, tab_fi, tab_predict = st.tabs([
    "📊 EDA",
    "🏋️ Model Performance",
    "🔍 Feature Importance",
    "🔮 Predict Salary",
])


# ── TAB 1: EDA ───────────────────────────────────────────────────────────────
with tab_eda:
    # KPI row
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        metric_card("Records", f"{len(filtered):,}")
    with col2:
        metric_card("Avg Salary", f"${filtered['salary_usd'].mean():,.0f}")
    with col3:
        metric_card("Median Salary", f"${filtered['salary_usd'].median():,.0f}")
    with col4:
        metric_card("Countries", f"{filtered['country'].nunique()}")
    with col5:
        metric_card("Job Roles", f"{filtered['job_role'].nunique()}")

    st.markdown("---")

    # Row 1: Distribution + By Country
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Salary Distribution")
        fig = px.histogram(
            filtered, x="salary_usd", nbins=60, color_discrete_sequence=["#e94560"],
            template="plotly_dark", opacity=0.85,
        )
        fig.update_layout(
            xaxis_title="Salary (USD)", yaxis_title="Count",
            margin=dict(t=10, b=40, l=40, r=10), height=380,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Salary by Country")
        country_avg = (
            filtered.groupby("country")["salary_usd"]
            .median()
            .sort_values(ascending=True)
            .reset_index()
        )
        fig = px.bar(
            country_avg, x="salary_usd", y="country", orientation="h",
            color="salary_usd", color_continuous_scale="Plasma",
            template="plotly_dark",
        )
        fig.update_layout(
            xaxis_title="Median Salary (USD)", yaxis_title="",
            margin=dict(t=10, b=40, l=10, r=10), height=380,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Row 2: By Role + By Experience
    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Salary by Job Role")
        fig = px.box(
            filtered, x="job_role", y="salary_usd",
            color="job_role", template="plotly_dark",
            color_discrete_sequence=px.colors.qualitative.Bold,
        )
        fig.update_layout(
            xaxis_title="", yaxis_title="Salary (USD)", showlegend=False,
            margin=dict(t=10, b=80, l=40, r=10), height=400,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        fig.update_xaxes(tickangle=30)
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        st.subheader("Salary by Experience Level")
        fig = px.violin(
            filtered, x="experience_level", y="salary_usd",
            color="experience_level", box=True, template="plotly_dark",
            color_discrete_sequence=px.colors.qualitative.Vivid,
        )
        fig.update_layout(
            xaxis_title="", yaxis_title="Salary (USD)", showlegend=False,
            margin=dict(t=10, b=40, l=40, r=10), height=400,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Row 3: Correlation Heatmap
    st.subheader("Correlation Heatmap (Numerical Features)")
    num_cols = filtered.select_dtypes(include=[np.number]).columns.tolist()
    # Pick top correlated with salary
    corr = filtered[num_cols].corr()
    top_corr = corr["salary_usd"].abs().sort_values(ascending=False).head(15).index.tolist()
    corr_subset = filtered[top_corr].corr()

    fig = px.imshow(
        corr_subset, text_auto=".2f", aspect="auto",
        color_continuous_scale="RdBu_r", template="plotly_dark",
    )
    fig.update_layout(
        margin=dict(t=10, b=10, l=10, r=10), height=500,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)


# ── TAB 2: Model Performance ────────────────────────────────────────────────
with tab_perf:
    if not artifacts.get("loaded"):
        st.warning(
            "⚠️ ML artifacts not found. Run `python ml_pipeline.py` first to train models.",
            icon="⚠️",
        )
    else:
        metrics = artifacts["metrics"]

        st.subheader("Model Comparison")

        # Metrics table
        rows = []
        for name, m in metrics["models"].items():
            rows.append({
                "Model": name,
                "R²": m["r2"],
                "MAE": f"${m['mae']:,.2f}",
                "RMSE": f"${m['rmse']:,.2f}",
                "MAPE": f"{m['mape']}%",
                "CV R² (mean)": m.get("cv_r2_mean", "–"),
                "CV R² (std)": m.get("cv_r2_std", "–"),
            })
        metrics_df = pd.DataFrame(rows)

        # Highlight best
        st.dataframe(
            metrics_df.style.highlight_max(subset=["R²"], color="#1a472a"),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("---")

        # Best model badge
        best = metrics["best_model"]
        best_m = metrics["best_model_metrics"]
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("🏆 Best Model", best)
        with c2:
            metric_card("R² Score", f"{best_m['r2']:.4f}")
        with c3:
            metric_card("MAE", f"${best_m['mae']:,.0f}")
        with c4:
            metric_card("RMSE", f"${best_m['rmse']:,.0f}")

        st.markdown("---")

        # Bar chart comparison
        st.subheader("R² Score Comparison")
        model_names = list(metrics["models"].keys())
        r2_scores = [metrics["models"][n]["r2"] for n in model_names]
        mae_scores = [metrics["models"][n]["mae"] for n in model_names]

        fig = make_subplots(rows=1, cols=2, subplot_titles=("R² Score (higher is better)", "MAE (lower is better)"))

        fig.add_trace(
            go.Bar(
                x=model_names, y=r2_scores, name="R²",
                marker_color=["#e94560" if n == best else "#533483" for n in model_names],
                text=[f"{s:.4f}" for s in r2_scores], textposition="outside",
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Bar(
                x=model_names, y=mae_scores, name="MAE",
                marker_color=["#e94560" if n == best else "#533483" for n in model_names],
                text=[f"${s:,.0f}" for s in mae_scores], textposition="outside",
            ),
            row=1, col=2,
        )
        fig.update_layout(
            template="plotly_dark", showlegend=False, height=400,
            margin=dict(t=40, b=40, l=40, r=40),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption(f"🕐 Trained at: {metrics.get('trained_at', 'N/A')}")


# ── TAB 3: Feature Importance ────────────────────────────────────────────────
with tab_fi:
    if not artifacts.get("loaded"):
        st.warning("⚠️ ML artifacts not found. Run `python ml_pipeline.py` first.", icon="⚠️")
    else:
        fi = artifacts["feature_importance"]
        if fi.empty:
            st.info("Feature importance not available for the best model.")
        else:
            st.subheader("Top 20 Most Important Features")
            top = fi.head(20).sort_values("importance", ascending=True)

            fig = px.bar(
                top, x="importance_pct", y="feature", orientation="h",
                color="importance_pct", color_continuous_scale="Viridis",
                template="plotly_dark",
                text=top["importance_pct"].apply(lambda x: f"{x:.1f}%"),
            )
            fig.update_layout(
                xaxis_title="Importance (%)", yaxis_title="",
                margin=dict(t=10, b=40, l=10, r=10), height=600,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                coloraxis_showscale=False,
            )
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

            # Table
            with st.expander("📋 Full Feature Importance Table"):
                st.dataframe(fi, use_container_width=True, hide_index=True)


# ── TAB 4: Predict Salary ───────────────────────────────────────────────────
with tab_predict:
    if not artifacts.get("loaded"):
        st.warning("⚠️ ML artifacts not found. Run `python ml_pipeline.py` first.", icon="⚠️")
    else:
        st.subheader("🔮 Salary Prediction")
        st.markdown("Fill in job parameters to get a predicted salary estimate.")

        model = artifacts["model"]
        scaler = artifacts["scaler"]
        label_encoders = artifacts["label_encoders"]
        col_info = artifacts["columns"]
        feature_names = col_info["features"]

        # Build input form
        with st.form("prediction_form"):
            cols = st.columns(3)

            input_vals = {}
            for i, feat in enumerate(feature_names):
                col_idx = i % 3
                with cols[col_idx]:
                    if feat in label_encoders:
                        le = label_encoders[feat]
                        options = list(le.classes_)
                        selected = st.selectbox(feat.replace("_", " ").title(), options, key=f"pred_{feat}")
                        input_vals[feat] = le.transform([selected])[0]
                    else:
                        # Numerical — use median as default
                        med = float(df[feat].median()) if feat in df.columns else 0.0
                        mn = float(df[feat].min()) if feat in df.columns else 0.0
                        mx = float(df[feat].max()) if feat in df.columns else 100.0
                        val = st.number_input(
                            feat.replace("_", " ").title(),
                            min_value=mn, max_value=mx, value=med,
                            key=f"pred_{feat}",
                        )
                        input_vals[feat] = val

            submitted = st.form_submit_button("⚡ Predict Salary", use_container_width=True)

        if submitted:
            # Build feature array
            input_df = pd.DataFrame([input_vals])

            # Scale numerical features
            num_features = [f for f in feature_names if f not in label_encoders]
            num_in_features = [f for f in num_features if f in input_df.columns]
            if num_in_features:
                input_df[num_in_features] = scaler.transform(input_df[num_in_features])

            prediction = model.predict(input_df[feature_names])[0]

            st.markdown("---")
            c1, c2, c3 = st.columns([1, 2, 1])
            with c2:
                st.success(f"### 💰 Predicted Salary: **${prediction:,.0f}** USD", icon="✅")

            # Confidence context
            st.markdown("---")
            st.subheader("📈 Prediction Context")
            pc1, pc2, pc3 = st.columns(3)
            with pc1:
                pctile = (df["salary_usd"] < prediction).mean() * 100
                metric_card("Salary Percentile", f"{pctile:.0f}th")
            with pc2:
                diff_from_mean = prediction - df["salary_usd"].mean()
                sign = "+" if diff_from_mean > 0 else ""
                metric_card("vs. Average", f"{sign}${diff_from_mean:,.0f}")
            with pc3:
                diff_from_median = prediction - df["salary_usd"].median()
                sign = "+" if diff_from_median > 0 else ""
                metric_card("vs. Median", f"{sign}${diff_from_median:,.0f}")


# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="text-align:center; color:#666; font-size:0.8rem;">'
    "Built with Streamlit • scikit-learn • Plotly | Global AI Jobs Dataset"
    "</div>",
    unsafe_allow_html=True,
)
