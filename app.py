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

# Set the page configuration
st.set_page_config(page_title="Universal AutoML Platform", page_icon="⚙️", layout="wide", initial_sidebar_state="expanded")

# --- CUSTOM CSS INJECTION ---
st.markdown("""
<style>
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    .stButton>button {
        border-radius: 8px;
        transition: all 0.3s ease-in-out;
        border: none;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.2);
    }
    [data-testid="stSidebar"] {
        background-color: #0f2027;
        border-right: 1px solid #2c5364;
    }
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    .streamlit-expanderHeader {
        background-color: #f0f2f6;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# --- 1. THE HERO BANNER ---
st.markdown("""
    <div style="background: linear-gradient(90deg, #0f2027 0%, #203a43 50%, #2c5364 100%); 
                padding: 30px; border-radius: 10px; margin-bottom: 25px; color: white; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        <h1 style="color: white; margin: 0; font-size: 2.5em;">⚙️ Universal AutoML Engine</h1>
        <p style="font-size: 1.2em; opacity: 0.9; margin-top: 10px;">Intelligent Data Pipelines & Predictive Analytics</p>
    </div>
""", unsafe_allow_html=True)

# --- 2. THE CONTROL SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Control Panel")
    st.markdown("Upload your dataset to begin the automated machine learning pipeline.")
    uploaded_file = st.file_uploader("📂 Upload CSV File", type=["csv"], key="train_data")

    # --- SMART CACHE CLEARING ---
    if uploaded_file is not None:
        if "last_uploaded" not in st.session_state or st.session_state.last_uploaded != uploaded_file.name:
            st.session_state.last_uploaded = uploaded_file.name
            for f in ["outputs/best_model.pkl", "outputs/performance_report.csv", "outputs/target_encoder.pkl"]:
                if os.path.exists(f):
                    os.remove(f)

# --- 3. MAIN DASHBOARD AREA ---
if uploaded_file is None:
    st.info("👈 Please upload a CSV file in the sidebar to get started.")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div style="background:#1a3a4a; padding:20px; border-radius:10px; border-left: 4px solid #3498db;">
            <h3 style="color:white;">1️⃣ Ingest</h3>
            <p style="color:#ccc;">Upload raw, messy data. The engine automatically handles missing values and formats.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style="background:#1a3a4a; padding:20px; border-radius:10px; border-left: 4px solid #2ecc71;">
            <h3 style="color:white;">2️⃣ Train</h3>
            <p style="color:#ccc;">Our algorithms dynamically detect task types and train multiple models simultaneously.</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div style="background:#1a3a4a; padding:20px; border-radius:10px; border-left: 4px solid #e74c3c;">
            <h3 style="color:white;">3️⃣ Predict</h3>
            <p style="color:#ccc;">Deploy the winning model instantly for single-record or bulk batch predictions.</p>
        </div>
        """, unsafe_allow_html=True)

else:
    raw_df = pd.read_csv(uploaded_file)

    with st.container():
        st.write(f"### 📊 Dataset Preview ({raw_df.shape[0]} rows, {raw_df.shape[1]} columns)")
        st.dataframe(raw_df.head(), use_container_width=True)

    with st.sidebar:
        st.divider()
        st.subheader("🎯 Configure Target")
        target_col = st.selectbox("What should the AI predict?", options=raw_df.columns, index=len(raw_df.columns)-1)

    # --- VISUAL DATA EXPLORER ---
    st.subheader("🔍 Exploratory Data Analysis")
    with st.expander("Open Interactive Data Visualizer"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Distribution of '{target_col}'**")
            fig_dist, ax_dist = plt.subplots(figsize=(6, 4))
            sns.histplot(raw_df[target_col], kde=True, color='#3498db', ax=ax_dist)
            st.pyplot(fig_dist)
        with col2:
            st.markdown("**Correlation Heatmap**")
            numeric_df = raw_df.select_dtypes(include=['number'])
            if not numeric_df.empty and len(numeric_df.columns) > 1:
                fig_corr, ax_corr = plt.subplots(figsize=(6, 4))
                sns.heatmap(numeric_df.corr(), annot=False, cmap='coolwarm', ax=ax_corr)
                st.pyplot(fig_corr)
            else:
                st.info("Not enough numeric columns to generate a heatmap.")

    st.divider()

    # --- STEP 2: RUN PIPELINE ---
    st.subheader("🚀 Execute AI Training Pipeline")
    if st.button("Start Automated Machine Learning", type="primary"):
        with st.spinner('Analyzing task type, cleaning data, and training algorithms. Please wait...'):
            try:
                cleaner = DataCleaner(raw_df)
                cleaned_df = cleaner.clean()

                preprocessor = Preprocessor(cleaned_df, target_column=target_col)
                X, y = preprocessor.process()

                # ✅ FIX: Save the target encoder so we can decode predictions later
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

                st.success("✅ Pipeline executed successfully! Scroll down for results.")
            except Exception as e:
                st.error(f"An error occurred: {e}")

    st.divider()

    # --- STEP 3: DISPLAY RESULTS ---
    report_path = "outputs/performance_report.csv"
    model_path = "outputs/best_model.pkl"
    encoder_path = "outputs/target_encoder.pkl"

    if os.path.exists(report_path) and os.path.exists(model_path):
        st.header("📊 Model Performance Report")
        report = pd.read_csv(report_path)
        model = joblib.load(model_path)

        # ✅ FIX: Load target encoder if it exists
        target_encoder = None
        if os.path.exists(encoder_path):
            with open(encoder_path, "rb") as f:
                target_encoder = pickle.load(f)

        is_classification = 'Accuracy' in report.columns
        best_model_name = report.iloc[0]['Model']

        # Always capture original classes for fallback
        original_classes = sorted(raw_df[target_col].dropna().unique())

        # DOWNLOAD MODEL BUTTON
        with open(model_path, "rb") as f:
            st.download_button(
                label="📥 Export Trained AI Brain (.pkl)",
                data=f,
                file_name=f"best_model_{best_model_name.replace(' ', '_')}.pkl",
                mime="application/octet-stream"
            )

        st.write("")

        # --- TABS FOR CLEANER UI ---
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Results", "📈 Visuals & Explainability", "🔮 Single Predict", "📦 Batch Predict"])

        with tab1:
            if is_classification:
                best_score = report.iloc[0]['Accuracy']
                col1, col2, col3 = st.columns(3)
                col1.metric(label="🏆 Winning Algorithm", value=best_model_name)
                col2.metric(label="🎯 Accuracy Score", value=f"{best_score*100:.2f}%")
                col3.metric(label="📊 Total Models Tested", value=str(len(report)))
                st.dataframe(report.style.highlight_max(subset=['Accuracy', 'F1-Score'], color='#2ecc71'), use_container_width=True)
            else:
                best_score = report.iloc[0]['R2-Score']
                rmse_score = report.iloc[0]['RMSE (Error)']
                col1, col2, col3 = st.columns(3)
                col1.metric(label="🏆 Winning Algorithm", value=best_model_name)
                col2.metric(label="🎯 R2-Score", value=f"{best_score:.4f}")
                col3.metric(label="📉 RMSE (Error)", value=f"{rmse_score:,.2f}", delta="- lower is better", delta_color="inverse")
                st.dataframe(report.style.highlight_max(subset=['R2-Score'], color='#2ecc71').highlight_min(subset=['RMSE (Error)'], color='#2ecc71'), use_container_width=True)

            # Model comparison bar chart
            st.markdown("#### 📊 Model Comparison Chart")
            metric = 'Accuracy' if is_classification else 'R2-Score'
            if metric in report.columns:
                fig_bar, ax_bar = plt.subplots(figsize=(8, 4))
                sns.barplot(data=report, x=metric, y='Model', palette='viridis', ax=ax_bar)
                ax_bar.set_title(f"Model Comparison — {metric}")
                st.pyplot(fig_bar)

        with tab2:
            # Advanced Evaluation Visuals
            st.header("📈 Advanced Evaluation Visuals")
            try:
                prep_eval = Preprocessor(raw_df, target_column=target_col)
                X_eval, y_eval = prep_eval.process()
                y_pred_eval = model.predict(X_eval)

                fig_eval, ax_eval = plt.subplots(figsize=(7, 5))
                if is_classification:
                    cm = confusion_matrix(y_eval, y_pred_eval)
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_eval, cbar=False)
                    ax_eval.set_title("Confusion Matrix", fontsize=14)
                    ax_eval.set_xlabel("Predicted Label", fontsize=12)
                    ax_eval.set_ylabel("True Label", fontsize=12)
                else:
                    sns.scatterplot(x=y_eval, y=y_pred_eval, alpha=0.6, color='#3498db', edgecolor='k', ax=ax_eval)
                    min_val = min(y_eval.min(), y_pred_eval.min())
                    max_val = max(y_eval.max(), y_pred_eval.max())
                    ax_eval.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Perfect Prediction Line")
                    ax_eval.set_title("Actual vs. Predicted Values", fontsize=14)
                    ax_eval.set_xlabel("Actual", fontsize=12)
                    ax_eval.set_ylabel("Predicted", fontsize=12)
                    ax_eval.legend()
                st.pyplot(fig_eval)
            except Exception as e:
                st.warning(f"⚠️ Could not generate advanced visuals. Details: {e}")

            st.divider()

            # Explainability
            st.header("🧠 Explainable AI (Feature Importance)")
            st.markdown(f"**Why does {best_model_name} make its decisions?**")
            try:
                prep_explain = Preprocessor(raw_df, target_column=target_col)
                X_explain, _ = prep_explain.process()
                feature_names = X_explain.columns

                importances = None
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                elif hasattr(model, 'coef_'):
                    importances = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
                    importances = np.abs(importances)

                if importances is not None:
                    importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': importances
                    }).sort_values(by='Importance', ascending=False).head(10)

                    fig_imp, ax_imp = plt.subplots(figsize=(8, 4))
                    sns.barplot(data=importance_df, x='Importance', y='Feature', palette='magma', ax=ax_imp)
                    ax_imp.set_title("Top 10 Most Important Features")
                    st.pyplot(fig_imp)
                else:
                    st.info("ℹ️ This model does not expose feature importances.")
            except Exception as e:
                st.warning("⚠️ Could not generate feature importance chart.")

        with tab3:
            # --- SINGLE PREDICTION ---
            st.header("🔮 Single Record Prediction")
            st.markdown("Edit the values below, then click predict.")

            template_df = raw_df.drop(columns=[target_col]).iloc[[0]].copy()
            edited_df = st.data_editor(template_df, hide_index=True)

            if st.button("🧬 Predict Single Record", type="primary"):
                with st.spinner("Analyzing new data..."):
                    try:
                        edited_df[target_col] = raw_df[target_col].iloc[0]
                        combined_df = pd.concat([raw_df, edited_df], ignore_index=True)

                        prep_new = Preprocessor(combined_df, target_column=target_col)
                        X_new, _ = prep_new.process()

                        final_unseen_features = X_new.iloc[[-1]]
                        raw_prediction = model.predict(final_unseen_features)[0]

                        if is_classification:
                            # ✅ FIX: Use inverse_transform to get original label back
                            if target_encoder is not None:
                                final_output = target_encoder.inverse_transform([int(raw_prediction)])[0]
                            else:
                                try:
                                    final_output = str(original_classes[int(raw_prediction)])
                                except (IndexError, ValueError):
                                    final_output = str(raw_prediction)

                            st.success(f"### 🚨 Predicted Category: **{final_output}**")

                            # Confidence scores
                            if hasattr(model, 'predict_proba'):
                                try:
                                    proba = model.predict_proba(final_unseen_features)[0]
                                    st.markdown("**Confidence Breakdown:**")
                                    # Get display labels
                                    if target_encoder is not None:
                                        class_labels = target_encoder.classes_
                                    else:
                                        class_labels = [str(c) for c in original_classes]
                                    for cls, prob in zip(class_labels, proba):
                                        st.progress(float(prob), text=f"{cls}: {prob*100:.1f}%")
                                except Exception:
                                    pass
                        else:
                            st.success(f"### 📈 Predicted Value: **{raw_prediction:,.2f}**")
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")

        with tab4:
            # --- BULK BATCH PREDICTION ---
            st.header("📦 Bulk Batch Prediction")
            st.markdown("Upload a CSV with the same columns as your training data (without the target column).")

            batch_file = st.file_uploader("Upload Batch CSV for Predictions", type=["csv"], key="batch_data")

            if batch_file is not None:
                batch_df = pd.read_csv(batch_file)
                st.info(f"Loaded {len(batch_df)} rows for batch prediction.")

                if st.button("🚀 Run Batch Prediction", type="primary"):
                    with st.spinner("Processing batch data..."):
                        try:
                            batch_process_df = batch_df.copy()
                            if target_col not in batch_process_df.columns:
                                batch_process_df[target_col] = raw_df[target_col].iloc[0]

                            combined_batch_df = pd.concat([raw_df, batch_process_df], ignore_index=True)

                            prep_batch = Preprocessor(combined_batch_df, target_column=target_col)
                            X_batch, _ = prep_batch.process()

                            final_batch_features = X_batch.iloc[-len(batch_df):]
                            batch_predictions = model.predict(final_batch_features)

                            if is_classification:
                                # ✅ FIX: Use inverse_transform for batch too
                                if target_encoder is not None:
                                    formatted_preds = target_encoder.inverse_transform([int(p) for p in batch_predictions])
                                else:
                                    try:
                                        formatted_preds = [str(original_classes[int(p)]) for p in batch_predictions]
                                    except (IndexError, ValueError):
                                        formatted_preds = [str(p) for p in batch_predictions]
                            else:
                                formatted_preds = [round(float(p), 2) for p in batch_predictions]

                            final_batch_df = batch_df.copy()
                            final_batch_df[f'AI_Predicted_{target_col}'] = formatted_preds

                            st.success("✅ Batch Prediction Complete!")
                            st.dataframe(final_batch_df.head(10), use_container_width=True)

                            csv_data = final_batch_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="📥 Download Full Predictions as CSV",
                                data=csv_data,
                                file_name="ai_batch_predictions.csv",
                                mime="text/csv"
                            )
                        except Exception as e:
                            st.error(f"⚠️ Batch prediction failed. Make sure your file matches the original data structure! Details: {e}")
