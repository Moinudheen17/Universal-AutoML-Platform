import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import pandas as pd
import joblib
import pickle
import os
import glob
from sklearn.metrics import confusion_matrix
from modules.data_cleaner import DataCleaner
from modules.preprocessor import Preprocessor
from modules.model_trainer import ModelTrainer
from modules.pdf_report import generate_pdf_report

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

st.set_page_config(page_title="Universal AutoML Engine", page_icon="⚙️", layout="wide", initial_sidebar_state="expanded")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
footer {visibility: hidden;}
#MainMenu {visibility: hidden;}
html, body, [data-testid="stAppViewContainer"] { background: #060b18 !important; }
[data-testid="stSidebar"] { background: #0a0f1e !important; border-right: 1px solid rgba(0,229,194,0.08) !important; }
[data-testid="stSidebar"] * { color: #ffffff !important; }
[data-testid="stSidebar"] .stFileUploader { border: 1px dashed rgba(0,229,194,0.2) !important; border-radius: 14px !important; background: rgba(0,229,194,0.02) !important; padding: 8px; }
[data-testid="stSidebar"] label { color: rgba(255,255,255,0.35) !important; font-size: 11px !important; }
[data-testid="stMain"] { background: #060b18 !important; }
.main .block-container { background: #060b18 !important; padding-top: 1.5rem !important; }
.stButton > button { background: linear-gradient(135deg, #00c9a7, #0077ff) !important; color: #fff !important; border: none !important; border-radius: 14px !important; padding: 10px 22px !important; font-size: 13px !important; font-weight: 500 !important; box-shadow: 0 4px 24px rgba(0,119,255,0.25) !important; transition: all 0.25s ease !important; }
.stButton > button:hover { box-shadow: 0 6px 32px rgba(0,229,194,0.35) !important; transform: translateY(-1px) !important; }
[data-testid="stDownloadButton"] button { background: rgba(0,229,194,0.08) !important; color: #00e5c2 !important; border: 1px solid rgba(0,229,194,0.25) !important; border-radius: 12px !important; }
[data-testid="stMetric"] { background: rgba(255,255,255,0.02) !important; border: 1px solid rgba(255,255,255,0.05) !important; border-radius: 18px !important; padding: 16px 20px !important; }
[data-testid="stMetricValue"] { color: #00e5c2 !important; font-size: 26px !important; }
[data-testid="stMetricLabel"] { color: rgba(255,255,255,0.3) !important; font-size: 11px !important; }
.stTabs [data-baseweb="tab-list"] { background: rgba(255,255,255,0.02) !important; border: 1px solid rgba(255,255,255,0.05) !important; border-radius: 14px !important; padding: 4px !important; gap: 4px !important; }
.stTabs [data-baseweb="tab"] { border-radius: 10px !important; color: rgba(255,255,255,0.35) !important; font-size: 12px !important; padding: 7px 16px !important; }
.stTabs [aria-selected="true"] { background: rgba(0,229,194,0.1) !important; color: #00e5c2 !important; border: 1px solid rgba(0,229,194,0.2) !important; }
.stTabs [data-baseweb="tab-border"] { display: none !important; }
.streamlit-expanderHeader { background: rgba(255,255,255,0.02) !important; border: 1px solid rgba(255,255,255,0.05) !important; border-radius: 14px !important; color: rgba(255,255,255,0.6) !important; }
[data-testid="stDataFrame"] { border-radius: 14px !important; overflow: hidden !important; border: 1px solid rgba(255,255,255,0.05) !important; }
[data-testid="stDataEditor"] { border-radius: 14px !important; border: 1px solid rgba(0,229,194,0.12) !important; overflow: hidden !important; }
.stProgress > div > div { background: linear-gradient(90deg, #00b894, #00e5c2) !important; border-radius: 4px !important; }
.stProgress > div { background: rgba(255,255,255,0.05) !important; border-radius: 4px !important; }
.stSuccess { background: rgba(0,229,194,0.07) !important; border: 1px solid rgba(0,229,194,0.2) !important; border-radius: 14px !important; color: #00e5c2 !important; }
.stInfo { background: rgba(0,119,255,0.07) !important; border: 1px solid rgba(0,119,255,0.2) !important; border-radius: 14px !important; }
.stWarning { background: rgba(245,158,11,0.07) !important; border: 1px solid rgba(245,158,11,0.2) !important; border-radius: 14px !important; }
.stError { background: rgba(248,113,113,0.07) !important; border: 1px solid rgba(248,113,113,0.2) !important; border-radius: 14px !important; }
[data-testid="stSelectbox"] > div > div { background: rgba(255,255,255,0.03) !important; border: 1px solid rgba(255,255,255,0.07) !important; border-radius: 12px !important; color: rgba(255,255,255,0.7) !important; }
hr { border: none !important; border-top: 1px solid rgba(255,255,255,0.05) !important; margin: 1.2rem 0 !important; }
h1, h2, h3, h4 { color: #ffffff !important; }
p, li, span { color: rgba(255,255,255,0.65); }
.stSpinner > div { border-top-color: #00e5c2 !important; }
</style>
""", unsafe_allow_html=True)

# ── HERO ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(120deg,rgba(0,229,194,0.07) 0%,rgba(0,119,255,0.07) 60%,rgba(139,92,246,0.05) 100%);
            border:1px solid rgba(255,255,255,0.06);border-radius:22px;padding:28px 32px;
            margin-bottom:8px;position:relative;overflow:hidden;">
  <div style="position:absolute;top:-60px;left:-60px;width:260px;height:260px;border-radius:50%;
              background:radial-gradient(circle,rgba(0,229,194,0.08),transparent 70%);pointer-events:none;"></div>
  <div style="position:absolute;bottom:-80px;right:80px;width:220px;height:220px;border-radius:50%;
              background:radial-gradient(circle,rgba(0,119,255,0.07),transparent 70%);pointer-events:none;"></div>
  <div style="position:relative;z-index:1;display:flex;align-items:center;justify-content:space-between;">
    <div>
      <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">
        <div style="width:40px;height:40px;border-radius:13px;background:linear-gradient(135deg,#00e5c2,#0077ff);
                    display:flex;align-items:center;justify-content:center;
                    box-shadow:0 4px 20px rgba(0,229,194,0.3);font-size:18px;">⚙️</div>
        <h1 style="margin:0;font-size:22px;font-weight:600;color:#fff;">Universal AutoML Engine</h1>
      </div>
      <p style="margin:0;font-size:13px;color:rgba(255,255,255,0.35);padding-left:52px;">
        Intelligent Data Pipelines & Predictive Analytics
      </p>
    </div>
    <div style="display:flex;align-items:center;gap:7px;background:rgba(0,229,194,0.08);
                border:1px solid rgba(0,229,194,0.2);border-radius:30px;padding:7px 16px;
                font-size:11px;color:#00e5c2;white-space:nowrap;">
      <span style="width:6px;height:6px;border-radius:50%;background:#00e5c2;display:inline-block;
                   box-shadow:0 0 8px rgba(0,229,194,0.8);"></span>
      AI Platform v2
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── HELPER: DATA QUALITY SCORE ────────────────────────────────────────────────
def compute_data_quality(df, target_col):
    scores = {}
    total  = len(df)

    missing_pct   = df.isnull().mean().mean() * 100
    duplicate_pct = df.duplicated().sum() / total * 100

    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    skewed   = 0
    for c in num_cols:
        if c != target_col:
            try:
                if abs(df[c].skew()) > 2:
                    skewed += 1
            except Exception:
                pass
    skew_pct = (skewed / max(len(num_cols), 1)) * 100

    imbalance_score = 0
    if df[target_col].dtype == 'object' or df[target_col].nunique() <= 10:
        vc = df[target_col].value_counts(normalize=True)
        if len(vc) > 1:
            imbalance_score = (1 - vc.min()) * 100 * 0.5

    overall = max(0, 100 - missing_pct * 2 - duplicate_pct - skew_pct * 0.3 - imbalance_score * 0.2)

    return {
        "overall":    round(overall, 1),
        "missing":    round(missing_pct, 1),
        "duplicates": round(duplicate_pct, 1),
        "skewed":     skewed,
        "imbalance":  round(imbalance_score, 1),
    }

# ── SAMPLE DATASETS ───────────────────────────────────────────────────────────
def get_sample_datasets():
    titanic_url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    iris_url    = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
    house_url   = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
    return {
        "🚢 Titanic (Classification)": (titanic_url, "Survived"),
        "🌸 Iris Flowers (Classification)": (iris_url, "species"),
        "🏠 Boston Housing (Regression)": (house_url, "medv"),
    }

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:16px;">
      <div style="width:34px;height:34px;border-radius:11px;background:linear-gradient(135deg,#00e5c2,#0077ff);
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

    st.markdown('<div style="font-size:9px;color:rgba(255,255,255,0.25);letter-spacing:1.2px;margin:12px 0 8px;">OR TRY A SAMPLE DATASET</div>', unsafe_allow_html=True)
    sample_datasets = get_sample_datasets()
    sample_choice   = st.selectbox("Choose sample", ["— None —"] + list(sample_datasets.keys()), label_visibility="collapsed")

    if uploaded_file is not None:
        if "last_uploaded" not in st.session_state or st.session_state.last_uploaded != uploaded_file.name:
            st.session_state.last_uploaded = uploaded_file.name
            st.session_state.pop("sample_df", None)
            files_to_clear = (
                ["outputs/best_model.pkl", "outputs/performance_report.csv",
                 "outputs/target_encoder.pkl", "outputs/preprocessor.pkl",
                 "outputs/tuning_summary.pkl", "outputs/smote_applied.pkl"]
                + glob.glob("outputs/model_*.pkl")
            )
            for f in files_to_clear:
                if os.path.exists(f):
                    os.remove(f)

# ── RESOLVE DATA SOURCE ───────────────────────────────────────────────────────
sample_target = None
if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)
elif sample_choice != "— None —":
    url, sample_target = sample_datasets[sample_choice]
    if "sample_df" not in st.session_state or st.session_state.get("sample_name") != sample_choice:
        with st.spinner("Loading sample dataset..."):
            try:
                st.session_state.sample_df   = pd.read_csv(url)
                st.session_state.sample_name = sample_choice
                files_to_clear = (
                    ["outputs/best_model.pkl", "outputs/performance_report.csv",
                     "outputs/target_encoder.pkl", "outputs/preprocessor.pkl"]
                    + glob.glob("outputs/model_*.pkl")
                )
                for f in files_to_clear:
                    if os.path.exists(f):
                        os.remove(f)
            except Exception as e:
                st.error(f"Could not load sample: {e}")
    raw_df = st.session_state.get("sample_df", None)
else:
    raw_df = None

# ── LANDING ───────────────────────────────────────────────────────────────────
if raw_df is None:
    st.markdown('<div style="text-align:center;padding:16px 0 8px;"><p style="font-size:13px;color:rgba(255,255,255,0.3);">Upload a CSV or choose a sample dataset to begin</p></div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    cards = [
        ("📥", "Ingest", "0,119,255", "Upload raw data or pick a built-in demo dataset. Auto-handles missing values."),
        ("🧠", "Train", "0,229,194", "Detects task type. Trains XGBoost, LightGBM, Random Forest & more simultaneously."),
        ("⚡", "Predict", "245,158,11", "Deploy winning model instantly. Single or bulk predictions with confidence scores."),
    ]
    for col, (icon, title, rgb, desc) in zip([col1, col2, col3], cards):
        with col:
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.02);border:1px solid rgba({rgb},0.15);
                        border-radius:20px;padding:22px 18px;text-align:center;">
              <div style="font-size:28px;margin-bottom:12px;">{icon}</div>
              <h3 style="color:#fff;font-size:15px;margin-bottom:8px;">{title}</h3>
              <p style="color:rgba(255,255,255,0.3);font-size:12px;line-height:1.7;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

else:
    # ── DATASET PREVIEW ───────────────────────────────────────────────────────
    st.markdown(f'<div style="font-size:12px;color:rgba(255,255,255,0.3);margin-bottom:6px;letter-spacing:0.5px;">DATASET · {raw_df.shape[0]} rows · {raw_df.shape[1]} columns</div>', unsafe_allow_html=True)
    st.dataframe(raw_df.head(), use_container_width=True)

    with st.sidebar:
        st.markdown('<div style="height:1px;background:linear-gradient(90deg,transparent,rgba(255,255,255,0.06),transparent);margin:8px 0 16px;"></div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size:9px;color:rgba(255,255,255,0.25);letter-spacing:1.2px;margin-bottom:8px;">TARGET COLUMN</div>', unsafe_allow_html=True)
        default_idx = list(raw_df.columns).index(sample_target) if sample_target and sample_target in raw_df.columns else len(raw_df.columns) - 1
        target_col  = st.selectbox("What should the AI predict?", options=raw_df.columns, index=default_idx)

    # ── DATA QUALITY SCORE ────────────────────────────────────────────────────
    dq = compute_data_quality(raw_df, target_col)
    score_color = "#00e5c2" if dq["overall"] >= 80 else "#f59e0b" if dq["overall"] >= 60 else "#f87171"
    st.markdown(f"""
    <div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.05);
                border-radius:18px;padding:18px 22px;margin:12px 0;">
      <div style="font-size:10px;color:rgba(255,255,255,0.25);letter-spacing:1px;margin-bottom:14px;">DATA QUALITY SCORE</div>
      <div style="display:flex;align-items:center;gap:24px;flex-wrap:wrap;">
        <div style="text-align:center;">
          <div style="font-size:36px;font-weight:600;color:{score_color};">{dq['overall']}%</div>
          <div style="font-size:10px;color:rgba(255,255,255,0.25);">Overall Health</div>
        </div>
        <div style="flex:1;display:grid;grid-template-columns:repeat(2,1fr);gap:10px;">
          <div style="background:rgba(255,255,255,0.02);border-radius:12px;padding:10px 14px;">
            <div style="font-size:10px;color:rgba(255,255,255,0.25);margin-bottom:4px;">Missing Values</div>
            <div style="font-size:16px;font-weight:500;color:{'#f87171' if dq['missing']>10 else '#00e5c2'};">{dq['missing']}%</div>
          </div>
          <div style="background:rgba(255,255,255,0.02);border-radius:12px;padding:10px 14px;">
            <div style="font-size:10px;color:rgba(255,255,255,0.25);margin-bottom:4px;">Duplicate Rows</div>
            <div style="font-size:16px;font-weight:500;color:{'#f87171' if dq['duplicates']>5 else '#00e5c2'};">{dq['duplicates']}%</div>
          </div>
          <div style="background:rgba(255,255,255,0.02);border-radius:12px;padding:10px 14px;">
            <div style="font-size:10px;color:rgba(255,255,255,0.25);margin-bottom:4px;">Skewed Columns</div>
            <div style="font-size:16px;font-weight:500;color:{'#f59e0b' if dq['skewed']>3 else '#00e5c2'};">{dq['skewed']}</div>
          </div>
          <div style="background:rgba(255,255,255,0.02);border-radius:12px;padding:10px 14px;">
            <div style="font-size:10px;color:rgba(255,255,255,0.25);margin-bottom:4px;">Class Imbalance</div>
            <div style="font-size:16px;font-weight:500;color:{'#f59e0b' if dq['imbalance']>30 else '#00e5c2'};">{dq['imbalance']}%</div>
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── EDA ───────────────────────────────────────────────────────────────────
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

    # ── PIPELINE ──────────────────────────────────────────────────────────────
    st.markdown('<div style="font-size:10px;color:rgba(255,255,255,0.2);letter-spacing:1.2px;margin-bottom:10px;">PIPELINE</div>', unsafe_allow_html=True)

    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        enable_tuning = st.checkbox("⚡ Auto-tune hyperparameters", value=True,
                                     help="Runs RandomizedSearchCV on each model. Slightly slower but more accurate.")
    with col_opt2:
        enable_smote = st.checkbox("⚖️ Auto-balance classes (SMOTE)", value=True,
                                    help="Oversamples minority class if imbalance detected. Classification only.")

    if st.button("🚀 Start Automated Machine Learning", type="primary"):
        prog = st.progress(0, text="Cleaning data...")
        try:
            cleaner    = DataCleaner(raw_df)
            cleaned_df = cleaner.clean()
            prog.progress(15, text="Preprocessing features...")

            preprocessor = Preprocessor(cleaned_df, target_column=target_col)
            X, y = preprocessor.process()
            prog.progress(30, text="Saving preprocessor...")

            os.makedirs("outputs", exist_ok=True)
            target_encoder = preprocessor.get_target_encoder()
            if target_encoder is not None:
                with open("outputs/target_encoder.pkl", "wb") as f:
                    pickle.dump(target_encoder, f)
            with open("outputs/preprocessor.pkl", "wb") as f:
                pickle.dump(preprocessor, f)

            # Column report
            col_report   = preprocessor.get_column_report()
            text_cols    = col_report["text_columns"]
            cat_cols     = [c for c in col_report["categorical_columns"] if c != target_col]
            tfidf_info   = col_report["tfidf_features"]
            dropped_cols = col_report.get("dropped_columns", [])
            if text_cols or cat_cols or dropped_cols:
                rh = '<div style="background:rgba(0,119,255,0.05);border:1px solid rgba(0,119,255,0.15);border-radius:14px;padding:14px 18px;margin-bottom:12px;">'
                rh += '<div style="font-size:10px;color:rgba(255,255,255,0.25);letter-spacing:1px;margin-bottom:10px;">COLUMN HANDLING REPORT</div>'
                rh += '<div style="display:flex;flex-wrap:wrap;gap:8px;">'
                for col in text_cols:
                    feat = tfidf_info.get(col, "TF-IDF")
                    rh += f'<span style="background:rgba(139,92,246,0.12);border:1px solid rgba(139,92,246,0.25);border-radius:20px;padding:4px 12px;font-size:11px;color:#c4b5fd;">📝 {col} → TF-IDF ({feat})</span>'
                for col in cat_cols:
                    rh += f'<span style="background:rgba(0,229,194,0.08);border:1px solid rgba(0,229,194,0.2);border-radius:20px;padding:4px 12px;font-size:11px;color:#00e5c2;">🏷️ {col} → Label Encoded</span>'
                for col in dropped_cols:
                    rh += f'<span style="background:rgba(248,113,113,0.08);border:1px solid rgba(248,113,113,0.2);border-radius:20px;padding:4px 12px;font-size:11px;color:#f87171;">🗑️ {col} → Dropped (noise)</span>'
                rh += '</div></div>'
                st.markdown(rh, unsafe_allow_html=True)

            prog.progress(45, text="Training models (XGBoost, LightGBM, Random Forest...)...")
            trainer    = ModelTrainer(X, y, enable_tuning=enable_tuning, enable_smote=enable_smote)
            results_df = trainer.train_and_evaluate()
            prog.progress(80, text="Saving models...")
            trainer.save_best_model("outputs")
            trainer.save_all_models("outputs")
            results_df.to_csv("outputs/performance_report.csv", index=False)

            # Save tuning summary + smote flag for display
            with open("outputs/tuning_summary.pkl", "wb") as f:
                pickle.dump(trainer.get_tuning_summary(), f)
            with open("outputs/smote_applied.pkl", "wb") as f:
                pickle.dump(trainer.smote_applied, f)

            prog.progress(100, text="Done!")
            st.success("✅ Pipeline complete! Results ready below.")

            if trainer.smote_applied:
                st.info("⚖️ Class imbalance detected — SMOTE oversampling was applied to balance training data.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

    st.markdown('<div style="height:1px;background:linear-gradient(90deg,transparent,rgba(255,255,255,0.06),transparent);margin:16px 0;"></div>', unsafe_allow_html=True)

    # ── RESULTS ───────────────────────────────────────────────────────────────
    report_path  = "outputs/performance_report.csv"
    model_path   = "outputs/best_model.pkl"
    encoder_path = "outputs/target_encoder.pkl"
    prep_path    = "outputs/preprocessor.pkl"

    if os.path.exists(report_path) and os.path.exists(model_path):
        report = pd.read_csv(report_path)
        model  = joblib.load(model_path)

        target_encoder     = None
        saved_preprocessor = None
        if os.path.exists(encoder_path):
            with open(encoder_path, "rb") as f:
                target_encoder = pickle.load(f)
        if os.path.exists(prep_path):
            with open(prep_path, "rb") as f:
                saved_preprocessor = pickle.load(f)

        is_classification = 'Accuracy' in report.columns
        best_model_name   = report.iloc[0]['Model']
        original_classes  = sorted(raw_df[target_col].dropna().unique())

        # ── MODEL SWITCHER ────────────────────────────────────────────────────
        st.markdown('<div style="font-size:10px;color:rgba(255,255,255,0.25);letter-spacing:1.2px;margin-bottom:8px;">MODEL SELECTION</div>', unsafe_allow_html=True)
        col_sel, col_info = st.columns([2, 1])
        with col_sel:
            selected_model_name = st.selectbox(
                "Choose model to use for predictions",
                options=report['Model'].tolist(), index=0,
                help="Auto-selected = best performing. Switch to compare any trained model."
            )
        selected_model_path = f"outputs/model_{selected_model_name.replace(' ', '_')}.pkl"
        if selected_model_name != best_model_name and os.path.exists(selected_model_path):
            model = joblib.load(selected_model_path)
            with col_info:
                st.markdown("<br>", unsafe_allow_html=True)
                st.info(f"Using: **{selected_model_name}**")
        else:
            with col_info:
                st.markdown("<br>", unsafe_allow_html=True)
                st.success(f"✅ Auto-best: **{best_model_name}**")

        st.markdown('<div style="height:1px;background:linear-gradient(90deg,transparent,rgba(255,255,255,0.06),transparent);margin:10px 0 14px;"></div>', unsafe_allow_html=True)

        active_model_path = selected_model_path if (selected_model_name != best_model_name and os.path.exists(selected_model_path)) else model_path
        with open(active_model_path, "rb") as f:
            st.download_button(
                label=f"📥 Export {selected_model_name} (.pkl)",
                data=f,
                file_name=f"model_{selected_model_name.replace(' ','_')}.pkl",
                mime="application/octet-stream"
            )

        st.write("")
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Results", "📈 Visuals & Explainability", "🔮 Single Predict", "📦 Batch Predict"])

        # ── TAB 1: RESULTS ────────────────────────────────────────────────────
        with tab1:
            if is_classification:
                best_score = report.iloc[0]['Accuracy']
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("🏆 Best Model",    best_model_name)
                c2.metric("🎯 Accuracy",      f"{best_score*100:.2f}%")
                c3.metric("📊 Models Tested", str(len(report)))
                if 'CV Mean' in report.columns:
                    cv_mean = report.iloc[0]['CV Mean']
                    cv_std  = report.iloc[0]['CV Std']
                    c4.metric("🔁 CV Score", f"{cv_mean*100:.1f}% ±{cv_std*100:.1f}%")
                st.write("")
                display_cols = [c for c in ['Model','Accuracy','F1-Score','CV Mean','CV Std'] if c in report.columns]
                fmt = report[display_cols].copy()
                if 'CV Mean' in fmt.columns:
                    fmt['CV Mean'] = fmt['CV Mean'].apply(lambda x: f"{x*100:.1f}%")
                if 'CV Std' in fmt.columns:
                    fmt['CV Std']  = fmt['CV Std'].apply(lambda x: f"±{x*100:.1f}%")
                st.dataframe(fmt, use_container_width=True)
            else:
                best_score = report.iloc[0]['R2-Score']
                rmse_score = report.iloc[0]['RMSE (Error)']
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("🏆 Best Model", best_model_name)
                c2.metric("🎯 R2-Score",   f"{best_score:.4f}")
                c3.metric("📉 RMSE",       f"{rmse_score:,.2f}", delta="lower is better", delta_color="inverse")
                if 'CV Mean' in report.columns:
                    cv_mean = report.iloc[0]['CV Mean']
                    cv_std  = report.iloc[0]['CV Std']
                    c4.metric("🔁 CV Score", f"{cv_mean:.3f} ±{cv_std:.3f}")
                st.write("")
                display_cols = [c for c in ['Model','R2-Score','RMSE (Error)','CV Mean','CV Std'] if c in report.columns]
                st.dataframe(report[display_cols], use_container_width=True)

            # Bar chart
            metric = 'Accuracy' if is_classification else 'R2-Score'
            if metric in report.columns:
                st.markdown('<div style="font-size:10px;color:rgba(255,255,255,0.2);letter-spacing:1px;margin:16px 0 8px;">MODEL COMPARISON</div>', unsafe_allow_html=True)
                fig_bar, ax_bar = plt.subplots(figsize=(8, max(3, len(report) * 0.6)))
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

            # CV comparison chart
            if 'CV Mean' in report.columns:
                st.markdown('<div style="font-size:10px;color:rgba(255,255,255,0.2);letter-spacing:1px;margin:16px 0 8px;">CROSS-VALIDATION SCORES (5-Fold)</div>', unsafe_allow_html=True)
                fig_cv, ax_cv = plt.subplots(figsize=(8, max(3, len(report) * 0.6)))
                fig_cv.patch.set_facecolor('#0d1325')
                ax_cv.set_facecolor('#0d1325')
                ax_cv.barh(report['Model'], report['CV Mean'],
                           xerr=report['CV Std'], color='#6366f1',
                           error_kw={'ecolor': '#a5b4fc', 'capsize': 4})
                ax_cv.set_title("CV Mean ± Std", color='#6b7280', fontsize=11)
                ax_cv.tick_params(labelcolor='#6b7280', color='#1e293b')
                ax_cv.spines[:].set_color('#1e293b')
                st.pyplot(fig_cv)

            # ── HYPERPARAMETER TUNING IMPROVEMENT ───────────────────────────────
            tuning_path = "outputs/tuning_summary.pkl"
            if os.path.exists(tuning_path):
                with open(tuning_path, "rb") as f:
                    tuning_summary = pickle.load(f)
                if tuning_summary and any(v.get("tuned") for v in tuning_summary.values()):
                    st.markdown('<div style="font-size:10px;color:rgba(255,255,255,0.2);letter-spacing:1px;margin:16px 0 8px;">⚡ HYPERPARAMETER TUNING IMPACT</div>', unsafe_allow_html=True)
                    rows = ""
                    for name, vals in tuning_summary.items():
                        if not vals.get("tuned"):
                            continue
                        before = vals["before"] * 100 if is_classification else vals["before"]
                        after  = vals["after"] * 100 if is_classification else vals["after"]
                        delta  = after - before
                        delta_color = "#00e5c2" if delta >= 0 else "#f87171"
                        unit = "%" if is_classification else ""
                        rows += f"""
                        <div style="display:flex;align-items:center;justify-content:space-between;
                                    padding:10px 16px;border-radius:12px;background:rgba(255,255,255,0.02);
                                    border:1px solid rgba(255,255,255,0.05);margin-bottom:6px;">
                          <span style="font-size:12px;color:rgba(255,255,255,0.6);">{name}</span>
                          <span style="font-size:12px;color:rgba(255,255,255,0.35);">{before:.2f}{unit} → {after:.2f}{unit}</span>
                          <span style="font-size:12px;font-weight:600;color:{delta_color};">{'+' if delta>=0 else ''}{delta:.2f}{unit}</span>
                        </div>"""
                    st.markdown(rows, unsafe_allow_html=True)

            # ── PDF EXPORT ───────────────────────────────────────────────────────
            st.markdown('<div style="height:1px;background:linear-gradient(90deg,transparent,rgba(255,255,255,0.06),transparent);margin:16px 0;"></div>', unsafe_allow_html=True)
            try:
                # Save charts to temp files for PDF embedding
                os.makedirs("outputs/charts", exist_ok=True)
                chart_paths = []
                fig_bar.savefig("outputs/charts/comparison.png", dpi=120, bbox_inches='tight', facecolor='#0d1325')
                chart_paths.append("outputs/charts/comparison.png")
                if 'CV Mean' in report.columns:
                    fig_cv.savefig("outputs/charts/cv_scores.png", dpi=120, bbox_inches='tight', facecolor='#0d1325')
                    chart_paths.append("outputs/charts/cv_scores.png")

                pdf_bytes = generate_pdf_report(
                    dataset_name=getattr(uploaded_file, 'name', sample_choice if 'sample_choice' in dir() else "dataset"),
                    task_type="Classification" if is_classification else "Regression",
                    best_model_name=best_model_name,
                    report_df=report,
                    target_col=target_col,
                    n_rows=raw_df.shape[0],
                    n_cols=raw_df.shape[1],
                    chart_image_paths=chart_paths
                )
                st.download_button(
                    label="📄 Export Full Report as PDF",
                    data=pdf_bytes,
                    file_name="automl_report.pdf",
                    mime="application/pdf"
                )
            except Exception as e:
                st.caption(f"PDF export unavailable: {e}")

        # ── TAB 2: VISUALS ────────────────────────────────────────────────────
        with tab2:
            st.markdown('<div style="font-size:10px;color:rgba(255,255,255,0.2);letter-spacing:1px;margin-bottom:12px;">EVALUATION VISUAL</div>', unsafe_allow_html=True)
            try:
                if saved_preprocessor:
                    X_eval = saved_preprocessor.transform(raw_df)
                else:
                    prep_eval = Preprocessor(raw_df, target_column=target_col)
                    X_eval, _ = prep_eval.process()
                y_eval = raw_df[target_col]
                if saved_preprocessor and saved_preprocessor.get_target_encoder():
                    y_eval = saved_preprocessor.get_target_encoder().transform(y_eval.astype(str).str.strip())
                y_pred_eval = model.predict(X_eval)
                fig_eval, ax_eval = plt.subplots(figsize=(7, 5))
                fig_eval.patch.set_facecolor('#0d1325')
                ax_eval.set_facecolor('#0d1325')
                if is_classification:
                    cm = confusion_matrix(y_eval, y_pred_eval)
                    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', ax=ax_eval,
                                cbar=False, linewidths=0.5, linecolor='#0d1325')
                    ax_eval.set_title("Confusion Matrix", color='#6b7280', fontsize=12)
                    ax_eval.set_xlabel("Predicted", color='#4b5563', fontsize=11)
                    ax_eval.set_ylabel("Actual",    color='#4b5563', fontsize=11)
                else:
                    ax_eval.scatter(y_eval, y_pred_eval, alpha=0.55, color='#00e5c2', edgecolors='#0d1325', s=40)
                    mn = min(y_eval.min(), y_pred_eval.min())
                    mx = max(y_eval.max(), y_pred_eval.max())
                    ax_eval.plot([mn, mx], [mn, mx], '--', color='#334155', lw=1.5, label="Perfect")
                    ax_eval.set_title("Actual vs Predicted", color='#6b7280', fontsize=12)
                    ax_eval.set_xlabel("Actual",    color='#4b5563')
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
                if saved_preprocessor:
                    X_explain = saved_preprocessor.transform(raw_df)
                else:
                    prep_explain = Preprocessor(raw_df, target_column=target_col)
                    X_explain, _ = prep_explain.process()
                feature_names = X_explain.columns
                importances   = None
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
                    sns.barplot(data=imp_df, x='Importance', y='Feature', palette='cool', ax=ax_imp)
                    ax_imp.set_title("Top 10 Features", color='#6b7280', fontsize=11)
                    ax_imp.tick_params(labelcolor='#6b7280', color='#1e293b')
                    ax_imp.spines[:].set_color('#1e293b')
                    ax_imp.set_xlabel("Importance", color='#4b5563')
                    ax_imp.set_ylabel("")
                    st.pyplot(fig_imp)
                else:
                    st.info("ℹ️ This model does not expose feature importances.")
            except Exception:
                st.warning("⚠️ Could not generate feature importance chart.")

            # ── SHAP EXPLAINABILITY ──────────────────────────────────────────────
            st.markdown('<div style="height:1px;background:linear-gradient(90deg,transparent,rgba(255,255,255,0.06),transparent);margin:16px 0;"></div>', unsafe_allow_html=True)
            st.markdown('<div style="font-size:10px;color:rgba(255,255,255,0.2);letter-spacing:1px;margin-bottom:4px;">SHAP EXPLAINABILITY</div>', unsafe_allow_html=True)
            st.markdown('<p style="font-size:11px;color:rgba(255,255,255,0.25);margin-bottom:10px;">Shows how much each feature pushes predictions higher or lower, across all samples.</p>', unsafe_allow_html=True)

            if not SHAP_AVAILABLE:
                st.caption("SHAP not installed — add `shap` to requirements.txt to enable this.")
            else:
                with st.expander("🧠 Show SHAP Summary Plot (may take a few seconds)"):
                    try:
                        if saved_preprocessor:
                            X_shap = saved_preprocessor.transform(raw_df)
                        else:
                            X_shap = X_explain

                        # Sample for speed on larger datasets
                        X_shap_sample = X_shap.sample(min(100, len(X_shap)), random_state=42) if len(X_shap) > 100 else X_shap

                        model_type = type(model).__name__
                        if any(t in model_type for t in ["RandomForest", "XGB", "LGBM", "GradientBoosting"]):
                            explainer   = shap.TreeExplainer(model)
                            shap_values = explainer.shap_values(X_shap_sample)
                        else:
                            background  = shap.sample(X_shap_sample, min(30, len(X_shap_sample)))
                            explainer   = shap.KernelExplainer(model.predict, background)
                            shap_values = explainer.shap_values(X_shap_sample, nsamples=50)

                        # Handle multi-class (list of arrays) vs binary/regression (single array)
                        if isinstance(shap_values, list):
                            shap_vals_plot = shap_values[1] if len(shap_values) > 1 else shap_values[0]
                        else:
                            shap_vals_plot = shap_values

                        fig_shap = plt.figure(figsize=(8, 5))
                        fig_shap.patch.set_facecolor('#0d1325')
                        shap.summary_plot(shap_vals_plot, X_shap_sample, show=False, plot_size=None)
                        ax_list = fig_shap.get_axes()
                        for ax in ax_list:
                            ax.set_facecolor('#0d1325')
                            ax.tick_params(labelcolor='#6b7280')
                            for spine in ax.spines.values():
                                spine.set_color('#1e293b')
                        st.pyplot(fig_shap)
                        st.caption("Each dot is one record. Red = high feature value, blue = low. Position shows impact on the prediction.")
                    except Exception as e:
                        st.warning(f"⚠️ Could not generate SHAP plot for this model/dataset combination. Details: {e}")

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
                        if saved_preprocessor:
                            X_new = saved_preprocessor.transform(edited_df)
                        else:
                            combined_df = pd.concat([raw_df, edited_df], ignore_index=True)
                            prep_new    = Preprocessor(combined_df, target_column=target_col)
                            X_new, _    = prep_new.process()
                            X_new       = X_new.iloc[[-1]]
                        raw_prediction = model.predict(X_new)[0]

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
                                    proba        = model.predict_proba(X_new)[0]
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
            st.markdown('<p style="font-size:12px;color:rgba(255,255,255,0.3);">Upload a CSV with the same columns as training data (without the target column).</p>', unsafe_allow_html=True)
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
                            if saved_preprocessor:
                                X_batch = saved_preprocessor.transform(batch_process_df)
                            else:
                                combined_batch   = pd.concat([raw_df, batch_process_df], ignore_index=True)
                                prep_batch       = Preprocessor(combined_batch, target_column=target_col)
                                X_batch, _       = prep_batch.process()
                                X_batch          = X_batch.iloc[-len(batch_df):]
                            preds = model.predict(X_batch)

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
                            st.error(f"⚠️ Batch prediction failed: {e}")
