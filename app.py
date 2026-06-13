import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import pandas as pd
import joblib
import pickle
import os
from sklearn.metrics import confusion_matrix
from modules.data_cleaner import DataCleaner
from modules.preprocessor import Preprocessor
from modules.model_trainer import ModelTrainer

st.set_page_config(page_title="Universal AutoML Engine", page_icon="⚙️", layout="wide", initial_sidebar_state="expanded")

# ── FLUID GLASSMORPHISM CSS ──────────────────────────────────────────────────
st.markdown("""
<style>
/* ── GLOBAL ── */
footer {visibility: hidden;}
#MainMenu {visibility: hidden;}
html, body, [data-testid="stAppViewContainer"] {
    background: #060b18 !important;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: #0a0f1e !important;
    border-right: 1px solid rgba(0,229,194,0.08) !important;
}
[data-testid="stSidebar"] * { color: #ffffff !important; }
[data-testid="stSidebar"] .stFileUploader {
    border: 1px dashed rgba(0,229,194,0.2) !important;
    border-radius: 14px !important;
    background: rgba(0,229,194,0.02) !important;
    padding: 8px;
}
[data-testid="stSidebar"] label { color: rgba(255,255,255,0.35) !important; font-size: 11px !important; }

/* ── MAIN BG ── */
[data-testid="stMain"] {
    background: #060b18 !important;
}
.main .block-container {
    background: #060b18 !important;
    padding-top: 1.5rem !important;
}

/* ── BUTTONS ── */
.stButton > button {
    background: linear-gradient(135deg, #00c9a7, #0077ff) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 10px 22px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    box-shadow: 0 4px 24px rgba(0,119,255,0.25) !important;
    transition: all 0.25s ease !important;
}
.stButton > button:hover {
    box-shadow: 0 6px 32px rgba(0,229,194,0.35) !important;
    transform: translateY(-1px) !important;
}

/* ── DOWNLOAD BUTTON ── */
[data-testid="stDownloadButton"] button {
    background: rgba(0,229,194,0.08) !important;
    color: #00e5c2 !important;
    border: 1px solid rgba(0,229,194,0.25) !important;
    border-radius: 12px !important;
}

/* ── METRICS ── */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.05) !important;
    border-radius: 18px !important;
    padding: 16px 20px !important;
}
[data-testid="stMetricValue"] { color: #00e5c2 !important; font-size: 26px !important; }
[data-testid="stMetricLabel"] { color: rgba(255,255,255,0.3) !important; font-size: 11px !important; }

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.05) !important;
    border-radius: 14px !important;
    padding: 4px !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px !important;
    color: rgba(255,255,255,0.35) !important;
    font-size: 12px !important;
    padding: 7px 16px !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(0,229,194,0.1) !important;
    color: #00e5c2 !important;
    border: 1px solid rgba(0,229,194,0.2) !important;
}
.stTabs [data-baseweb="tab-border"] { display: none !important; }

/* ── EXPANDER ── */
.streamlit-expanderHeader {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.05) !important;
    border-radius: 14px !important;
    color: rgba(255,255,255,0.6) !important;
}
.streamlit-expanderContent {
    background: rgba(255,255,255,0.01) !important;
    border: 1px solid rgba(255,255,255,0.04) !important;
    border-radius: 0 0 14px 14px !important;
}

/* ── DATAFRAME ── */
[data-testid="stDataFrame"] {
    border-radius: 14px !important;
    overflow: hidden !important;
    border: 1px solid rgba(255,255,255,0.05) !important;
}

/* ── DATA EDITOR ── */
[data-testid="stDataEditor"] {
    border-radius: 14px !important;
    border: 1px solid rgba(0,229,194,0.12) !important;
    overflow: hidden !important;
}

/* ── PROGRESS BARS ── */
.stProgress > div > div {
    background: linear-gradient(90deg, #00b894, #00e5c2) !important;
    border-radius: 4px !important;
}
.stProgress > div {
    background: rgba(255,255,255,0.05) !important;
    border-radius: 4px !important;
}

/* ── ALERTS ── */
.stSuccess {
    background: rgba(0,229,194,0.07) !important;
    border: 1px solid rgba(0,229,194,0.2) !important;
    border-radius: 14px !important;
    color: #00e5c2 !important;
}
.stInfo {
    background: rgba(0,119,255,0.07) !important;
    border: 1px solid rgba(0,119,255,0.2) !important;
    border-radius: 14px !important;
}
.stWarning {
    background: rgba(245,158,11,0.07) !important;
    border: 1px solid rgba(245,158,11,0.2) !important;
    border-radius: 14px !important;
}
.stError {
    background: rgba(248,113,113,0.07) !important;
    border: 1px solid rgba(248,113,113,0.2) !important;
    border-radius: 14px !important;
}

/* ── SELECTBOX ── */
[data-testid="stSelectbox"] > div > div {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 12px !important;
    color: rgba(255,255,255,0.7) !important;
}

/* ── DIVIDER ── */
hr {
    border: none !important;
    border-top: 1px solid rgba(255,255,255,0.05) !important;
    margin: 1.2rem 0 !important;
}

/* ── HEADINGS ── */
h1, h2, h3, h4 { color: #ffffff !important; }
p, li, span { color: rgba(255,255,255,0.65); }

/* ── SPINNER ── */
.stSpinner > div { border-top-color: #00e5c2 !important; }

/* ── MATPLOTLIB CHARTS ── */
.stPlotlyChart, .stPyplot {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.05) !important;
    border-radius: 18px !important;
    padding: 12px !important;
}
</style>
""", unsafe_allow_html=True)

# ── HERO BANNER ──────────────────────────────────────────────────────────────
st.markdown("""
<div style="
    background: linear-gradient(120deg, rgba(0,229,194,0.07) 0%, rgba(0,119,255,0.07) 60%, rgba(139,92,246,0.05) 100%);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 22px;
    padding: 28px 32px;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: relative;
    overflow: hidden;
">
  <div style="position:absolute;top:-60px;left:-60px;width:260px;height:260px;border-radius:50%;
              background:radial-gradient(circle,rgba(0,229,194,0.08),transparent 70%);pointer-events:none;"></div>
  <div style="position:absolute;bottom:-80px;right:80px;width:220px;height:220px;border-radius:50%;
              background:radial-gradient(circle,rgba(0,119,255,0.07),transparent 70%);pointer-events:none;"></div>
  <div style="position:relative;z-index:1;">
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">
      <div style="width:40px;height:40px;border-radius:13px;
                  background:linear-gradient(135deg,#00e5c2,#0077ff);
                  display:flex;align-items:center;justify-content:center;
                  box-shadow:0 4px 20px rgba(0,229,194,0.3);font-size:18px;">⚙️</div>
      <h1 style="margin:0;font-size:22px;font-weight:600;color:#fff;">Universal AutoML Engine</h1>
    </div>
    <p style="margin:0;font-size:13px;color:rgba(255,255,255,0.35);padding-left:52px;">
      Intelligent Data Pipelines & Predictive Analytics
    </p>
  </div>
  <div style="display:flex;align-items:center;gap:7px;
              background:rgba(0,229,194,0.08);
              border:1px solid rgba(0,229,194,0.2);
              border-radius:30px;padding:7px 16px;
              font-size:11px;color:#00e5c2;white-space:nowrap;position:relative;z-index:1;">
    <span style="width:6px;height:6px;border-radius:50%;background:#00e5c2;display:inline-block;
                 box-shadow:0 0 8px rgba(0,229,194,0.8);"></span>
    AI Platform v2
  </div>
</div>
""", unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:16px;">
      <div style="width:34px;height:34px;border-radius:11px;
                  background:linear-gradient(135deg,#00e5c2,#0077ff);
                  display:flex;align-items:center;justify-content:center;
                  box-shadow:0 4px 16px rgba(0,229,194,0.25);font-size:16px;">⚙️</div>
      <div>
        <div style="font-size:13px;font-weight:500;color:#fff;">Control Panel</div>
        <div style="font-size:10px;color:rgba(0,229,194,0.55);">AutoML Engine</div>
      </div>
    </div>
    <div style="height:1px;background:linear-gradient(90deg,transparent,rgba(255,255,255,0.06),transparent);margin-bottom:16px;"></div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📂 Upload CSV Dataset", type=["csv"], key="train_data")

    if uploaded_file is not None:
        if "last_uploaded" not in st.session_state or st.session_state.last_uploaded != uploaded_file.name:
            st.session_state.last_uploaded = uploaded_file.name
            for f in ["outputs/best_model.pkl", "outputs/performance_report.csv", "outputs/target_encoder.pkl"]:
                if os.path.exists(f):
                    os.remove(f)

# ── LANDING PAGE ──────────────────────────────────────────────────────────────
if uploaded_file is None:
    st.markdown("""
    <div style="text-align:center;padding:16px 0 8px;">
      <p style="font-size:13px;color:rgba(255,255,255,0.3);">
        Upload a CSV file in the sidebar to begin
      </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    cards = [
        ("📥", "Ingest", "0,119,255", "Upload raw, messy data. Auto-handles missing values and formats."),
        ("🧠", "Train", "0,229,194", "Detects task type automatically. Trains multiple models simultaneously."),
        ("⚡", "Predict", "245,158,11", "Deploy the winning model instantly for single or bulk predictions."),
    ]
    for col, (icon, title, rgb, desc) in zip([col1, col2, col3], cards):
        with col:
            st.markdown(f"""
            <div style="
                background: rgba(255,255,255,0.02);
                border: 1px solid rgba({rgb},0.15);
                border-radius: 20px;
                padding: 22px 18px;
                text-align: center;
                transition: all 0.25s;
            ">
              <div style="font-size:28px;margin-bottom:12px;">{icon}</div>
              <h3 style="color:#fff;font-size:15px;margin-bottom:8px;">{title}</h3>
              <p style="color:rgba(255,255,255,0.3);font-size:12px;line-height:1.7;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

else:
    raw_df = pd.read_csv(uploaded_file)

    # Dataset preview
    st.markdown(f"""
    <div style="font-size:12px;color:rgba(255,255,255,0.3);margin-bottom:6px;letter-spacing:0.5px;">
      DATASET · {raw_df.shape[0]} rows · {raw_df.shape[1]} columns
    </div>
    """, unsafe_allow_html=True)
    st.dataframe(raw_df.head(), use_container_width=True)

    with st.sidebar:
        st.markdown("""<div style="height:1px;background:linear-gradient(90deg,transparent,rgba(255,255,255,0.06),transparent);margin:8px 0 16px;"></div>""", unsafe_allow_html=True)
        st.markdown('<div style="font-size:9px;color:rgba(255,255,255,0.25);letter-spacing:1.2px;margin-bottom:8px;">TARGET COLUMN</div>', unsafe_allow_html=True)
        target_col = st.selectbox("What should the AI predict?", options=raw_df.columns, index=len(raw_df.columns)-1)

    # EDA
    st.markdown('<div style="height:1px;background:linear-gradient(90deg,transparent,rgba(255,255,255,0.06),transparent);margin:8px 0;"></div>', unsafe_allow_html=True)
    with st.expander("🔍 Exploratory Data Analysis"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f'<div style="font-size:12px;color:rgba(255,255,255,0.4);margin-bottom:8px;">Distribution of {target_col}</div>', unsafe_allow_html=True)
            fig_dist, ax_dist = plt.subplots(figsize=(6, 4))
            fig_dist.patch.set_facecolor('#0d1325')
            ax_dist.set_facecolor('#0d1325')
            sns.histplot(raw_df[target_col], kde=True, color='#00e5c2', ax=ax_dist)
            ax_dist.tick_params(labelcolor='#6b7280', color='#1e293b')
            ax_dist.spines[:].set_color('#1e293b')
            st.pyplot(fig_dist)
        with col2:
            st.markdown('<div style="font-size:12px;color:rgba(255,255,255,0.4);margin-bottom:8px;">Correlation Heatmap</div>', unsafe_allow_html=True)
            numeric_df = raw_df.select_dtypes(include=['number'])
            if not numeric_df.empty and len(numeric_df.columns) > 1:
                fig_corr, ax_corr = plt.subplots(figsize=(6, 4))
                fig_corr.patch.set_facecolor('#0d1325')
                ax_corr.set_facecolor('#0d1325')
                sns.heatmap(numeric_df.corr(), annot=False, cmap='cool', ax=ax_corr)
                ax_corr.tick_params(labelcolor='#6b7280', color='#1e293b')
                st.pyplot(fig_corr)
            else:
                st.info("Not enough numeric columns for a heatmap.")

    st.markdown('<div style="height:1px;background:linear-gradient(90deg,transparent,rgba(255,255,255,0.06),transparent);margin:16px 0;"></div>', unsafe_allow_html=True)

    # Run pipeline
    st.markdown("""
    <div style="font-size:10px;color:rgba(255,255,255,0.2);letter-spacing:1.2px;margin-bottom:10px;">
      PIPELINE
    </div>
    """, unsafe_allow_html=True)

    if st.button("🚀 Start Automated Machine Learning", type="primary"):
        with st.spinner("Analyzing task type, cleaning data, and training algorithms..."):
            try:
                cleaner = DataCleaner(raw_df)
                cleaned_df = cleaner.clean()

                preprocessor = Preprocessor(cleaned_df, target_column=target_col)
                X, y = preprocessor.process()

                target_encoder = preprocessor.get_target_encoder()
                if not os.path.exists("outputs"):
                    os.makedirs("outputs")
                if target_encoder is not None:
                    with open("outputs/target_encoder.pkl", "wb") as f:
                        pickle.dump(target_encoder, f)

                trainer = ModelTrainer(X, y)
                results_df = trainer.train_and_evaluate()
                trainer.save_best_model("outputs")
                results_df.to_csv("outputs/performance_report.csv", index=False)

                st.success("✅ Pipeline complete! Results ready below.")
            except Exception as e:
                st.error(f"An error occurred: {e}")

    st.markdown('<div style="height:1px;background:linear-gradient(90deg,transparent,rgba(255,255,255,0.06),transparent);margin:16px 0;"></div>', unsafe_allow_html=True)

    # ── RESULTS ──────────────────────────────────────────────────────────────
    report_path = "outputs/performance_report.csv"
    model_path  = "outputs/best_model.pkl"
    encoder_path = "outputs/target_encoder.pkl"

    if os.path.exists(report_path) and os.path.exists(model_path):
        report = pd.read_csv(report_path)
        model  = joblib.load(model_path)

        target_encoder = None
        if os.path.exists(encoder_path):
            with open(encoder_path, "rb") as f:
                target_encoder = pickle.load(f)

        is_classification = 'Accuracy' in report.columns
        best_model_name   = report.iloc[0]['Model']
        original_classes  = sorted(raw_df[target_col].dropna().unique())

        # Download button
        with open(model_path, "rb") as f:
            st.download_button(
                label="📥 Export Trained Model (.pkl)",
                data=f,
                file_name=f"best_model_{best_model_name.replace(' ','_')}.pkl",
                mime="application/octet-stream"
            )

        st.write("")

        tab1, tab2, tab3, tab4 = st.tabs(["📊 Results", "📈 Visuals & Explainability", "🔮 Single Predict", "📦 Batch Predict"])

        # ── TAB 1: RESULTS ───────────────────────────────────────────────────
        with tab1:
            if is_classification:
                best_score = report.iloc[0]['Accuracy']
                c1, c2, c3 = st.columns(3)
                c1.metric("🏆 Winning Algorithm", best_model_name)
                c2.metric("🎯 Accuracy", f"{best_score*100:.2f}%")
                c3.metric("📊 Models Tested", str(len(report)))
                st.write("")
                st.dataframe(
                    report.style.highlight_max(subset=['Accuracy','F1-Score'], color='rgba(0,229,194,0.15)'),
                    use_container_width=True
                )
            else:
                best_score = report.iloc[0]['R2-Score']
                rmse_score = report.iloc[0]['RMSE (Error)']
                c1, c2, c3 = st.columns(3)
                c1.metric("🏆 Winning Algorithm", best_model_name)
                c2.metric("🎯 R2-Score", f"{best_score:.4f}")
                c3.metric("📉 RMSE", f"{rmse_score:,.2f}", delta="lower is better", delta_color="inverse")
                st.write("")
                st.dataframe(
                    report.style.highlight_max(subset=['R2-Score'], color='rgba(0,229,194,0.15)')
                                .highlight_min(subset=['RMSE (Error)'], color='rgba(0,229,194,0.15)'),
                    use_container_width=True
                )

            # Bar chart
            metric = 'Accuracy' if is_classification else 'R2-Score'
            if metric in report.columns:
                st.markdown(f'<div style="font-size:10px;color:rgba(255,255,255,0.2);letter-spacing:1px;margin:16px 0 8px;">MODEL COMPARISON</div>', unsafe_allow_html=True)
                fig_bar, ax_bar = plt.subplots(figsize=(8, 3.5))
                fig_bar.patch.set_facecolor('#0d1325')
                ax_bar.set_facecolor('#0d1325')
                colors = ['#00e5c2' if i == 0 else '#1e3a5f' for i in range(len(report))]
                sns.barplot(data=report, x=metric, y='Model', palette=colors, ax=ax_bar)
                ax_bar.set_title(f"Model Comparison — {metric}", color='#6b7280', fontsize=11)
                ax_bar.tick_params(labelcolor='#6b7280', color='#1e293b')
                ax_bar.spines[:].set_color('#1e293b')
                ax_bar.set_xlabel(metric, color='#4b5563', fontsize=10)
                ax_bar.set_ylabel("")
                st.pyplot(fig_bar)

        # ── TAB 2: VISUALS ────────────────────────────────────────────────────
        with tab2:
            st.markdown('<div style="font-size:10px;color:rgba(255,255,255,0.2);letter-spacing:1px;margin-bottom:12px;">EVALUATION VISUAL</div>', unsafe_allow_html=True)
            try:
                prep_eval  = Preprocessor(raw_df, target_column=target_col)
                X_eval, y_eval = prep_eval.process()
                y_pred_eval = model.predict(X_eval)
                fig_eval, ax_eval = plt.subplots(figsize=(7, 5))
                fig_eval.patch.set_facecolor('#0d1325')
                ax_eval.set_facecolor('#0d1325')
                if is_classification:
                    cm = confusion_matrix(y_eval, y_pred_eval)
                    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', ax=ax_eval, cbar=False,
                                linewidths=0.5, linecolor='#0d1325')
                    ax_eval.set_title("Confusion Matrix", color='#6b7280', fontsize=12)
                    ax_eval.set_xlabel("Predicted", color='#4b5563', fontsize=11)
                    ax_eval.set_ylabel("Actual", color='#4b5563', fontsize=11)
                else:
                    ax_eval.scatter(y_eval, y_pred_eval, alpha=0.55, color='#00e5c2', edgecolors='#0d1325', s=40)
                    mn = min(y_eval.min(), y_pred_eval.min())
                    mx = max(y_eval.max(), y_pred_eval.max())
                    ax_eval.plot([mn, mx], [mn, mx], '--', color='#334155', lw=1.5, label="Perfect")
                    ax_eval.set_title("Actual vs Predicted", color='#6b7280', fontsize=12)
                    ax_eval.set_xlabel("Actual", color='#4b5563')
                    ax_eval.set_ylabel("Predicted", color='#4b5563')
                    ax_eval.legend(labelcolor='#6b7280', framealpha=0)
                ax_eval.tick_params(labelcolor='#6b7280', color='#1e293b')
                ax_eval.spines[:].set_color('#1e293b')
                st.pyplot(fig_eval)
            except Exception as e:
                st.warning(f"⚠️ Could not generate visuals. Details: {e}")

            st.markdown('<div style="height:1px;background:linear-gradient(90deg,transparent,rgba(255,255,255,0.06),transparent);margin:16px 0;"></div>', unsafe_allow_html=True)
            st.markdown('<div style="font-size:10px;color:rgba(255,255,255,0.2);letter-spacing:1px;margin-bottom:12px;">FEATURE IMPORTANCE</div>', unsafe_allow_html=True)
            try:
                prep_explain = Preprocessor(raw_df, target_column=target_col)
                X_explain, _ = prep_explain.process()
                feature_names = X_explain.columns
                importances = None
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                elif hasattr(model, 'coef_'):
                    importances = np.abs(model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_)
                if importances is not None:
                    imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}) \
                                .sort_values('Importance', ascending=False).head(10)
                    fig_imp, ax_imp = plt.subplots(figsize=(8, 4))
                    fig_imp.patch.set_facecolor('#0d1325')
                    ax_imp.set_facecolor('#0d1325')
                    colors_imp = [f'#{int(0 + (0)*i/9):02x}{int(184 + (229-184)*i/9):02x}{int(148 + (194-148)*i/9):02x}'
                                  for i in range(len(imp_df))]
                    sns.barplot(data=imp_df, x='Importance', y='Feature', palette='cool', ax=ax_imp)
                    ax_imp.set_title("Top Features", color='#6b7280', fontsize=11)
                    ax_imp.tick_params(labelcolor='#6b7280', color='#1e293b')
                    ax_imp.spines[:].set_color('#1e293b')
                    ax_imp.set_xlabel("Importance", color='#4b5563')
                    ax_imp.set_ylabel("")
                    st.pyplot(fig_imp)
                else:
                    st.info("ℹ️ This model does not expose feature importances.")
            except Exception:
                st.warning("⚠️ Could not generate feature importance chart.")

        # ── TAB 3: SINGLE PREDICT ─────────────────────────────────────────────
        with tab3:
            st.markdown('<div style="font-size:10px;color:rgba(255,255,255,0.2);letter-spacing:1px;margin-bottom:12px;">SINGLE RECORD PREDICTION</div>', unsafe_allow_html=True)
            st.markdown('<p style="font-size:12px;color:rgba(255,255,255,0.3);">Edit the values below then click Predict.</p>', unsafe_allow_html=True)
            template_df = raw_df.drop(columns=[target_col]).iloc[[0]].copy()
            edited_df   = st.data_editor(template_df, hide_index=True)

            if st.button("🧬 Predict Single Record", type="primary"):
                with st.spinner("Analyzing..."):
                    try:
                        edited_df[target_col] = raw_df[target_col].iloc[0]
                        combined_df = pd.concat([raw_df, edited_df], ignore_index=True)
                        prep_new = Preprocessor(combined_df, target_column=target_col)
                        X_new, _ = prep_new.process()
                        raw_prediction = model.predict(X_new.iloc[[-1]])[0]

                        if is_classification:
                            if target_encoder is not None:
                                final_output = target_encoder.inverse_transform([int(raw_prediction)])[0]
                            else:
                                try:
                                    final_output = str(original_classes[int(raw_prediction)])
                                except (IndexError, ValueError):
                                    final_output = str(raw_prediction)
                            st.markdown(f"""
                            <div style="background:rgba(0,229,194,0.06);border:1px solid rgba(0,229,194,0.2);
                                        border-radius:16px;padding:18px 22px;margin-top:12px;
                                        display:flex;align-items:center;justify-content:space-between;">
                              <span style="font-size:12px;color:rgba(255,255,255,0.35);">Prediction</span>
                              <span style="font-size:20px;font-weight:600;color:#00e5c2;">{final_output}</span>
                            </div>
                            """, unsafe_allow_html=True)
                            if hasattr(model, 'predict_proba'):
                                try:
                                    proba = model.predict_proba(X_new.iloc[[-1]])[0]
                                    class_labels = target_encoder.classes_ if target_encoder else [str(c) for c in original_classes]
                                    st.markdown('<div style="font-size:10px;color:rgba(255,255,255,0.2);letter-spacing:1px;margin:14px 0 8px;">CONFIDENCE</div>', unsafe_allow_html=True)
                                    for cls, prob in zip(class_labels, proba):
                                        st.progress(float(prob), text=f"{cls}: {prob*100:.1f}%")
                                except Exception:
                                    pass
                        else:
                            st.markdown(f"""
                            <div style="background:rgba(0,119,255,0.06);border:1px solid rgba(0,119,255,0.2);
                                        border-radius:16px;padding:18px 22px;margin-top:12px;
                                        display:flex;align-items:center;justify-content:space-between;">
                              <span style="font-size:12px;color:rgba(255,255,255,0.35);">Predicted Value</span>
                              <span style="font-size:20px;font-weight:600;color:#60a5fa;">{raw_prediction:,.2f}</span>
                            </div>
                            """, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")

        # ── TAB 4: BATCH PREDICT ──────────────────────────────────────────────
        with tab4:
            st.markdown('<div style="font-size:10px;color:rgba(255,255,255,0.2);letter-spacing:1px;margin-bottom:12px;">BULK BATCH PREDICTION</div>', unsafe_allow_html=True)
            st.markdown('<p style="font-size:12px;color:rgba(255,255,255,0.3);">Upload a CSV with the same columns as your training data (without the target column).</p>', unsafe_allow_html=True)
            batch_file = st.file_uploader("Upload Batch CSV", type=["csv"], key="batch_data")

            if batch_file is not None:
                batch_df = pd.read_csv(batch_file)
                st.info(f"Loaded {len(batch_df)} rows for batch prediction.")

                if st.button("🚀 Run Batch Prediction", type="primary"):
                    with st.spinner("Processing batch..."):
                        try:
                            batch_process_df = batch_df.copy()
                            if target_col not in batch_process_df.columns:
                                batch_process_df[target_col] = raw_df[target_col].iloc[0]
                            combined_batch = pd.concat([raw_df, batch_process_df], ignore_index=True)
                            prep_batch = Preprocessor(combined_batch, target_column=target_col)
                            X_batch, _ = prep_batch.process()
                            preds = model.predict(X_batch.iloc[-len(batch_df):])

                            if is_classification:
                                if target_encoder is not None:
                                    formatted = target_encoder.inverse_transform([int(p) for p in preds])
                                else:
                                    try:
                                        formatted = [str(original_classes[int(p)]) for p in preds]
                                    except (IndexError, ValueError):
                                        formatted = [str(p) for p in preds]
                            else:
                                formatted = [round(float(p), 2) for p in preds]

                            final_batch_df = batch_df.copy()
                            final_batch_df[f'AI_Predicted_{target_col}'] = formatted
                            st.success("✅ Batch prediction complete!")
                            st.dataframe(final_batch_df.head(10), use_container_width=True)
                            st.download_button(
                                label="📥 Download Full Predictions as CSV",
                                data=final_batch_df.to_csv(index=False).encode('utf-8'),
                                file_name="ai_batch_predictions.csv",
                                mime="text/csv"
                            )
                        except Exception as e:
                            st.error(f"⚠️ Batch prediction failed. Make sure your file matches the original structure! Details: {e}")
