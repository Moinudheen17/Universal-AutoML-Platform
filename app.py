import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import pandas as pd
import joblib
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
        background-color: #f8f9fa;
        border-right: 1px solid #e0e0e0;
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
            if os.path.exists("outputs/best_model.pkl"):
                os.remove("outputs/best_model.pkl")
            if os.path.exists("outputs/performance_report.csv"):
                os.remove("outputs/performance_report.csv")

# --- 3. MAIN DASHBOARD AREA ---
if uploaded_file is None:
    st.info("👈 Please upload a CSV file in the sidebar to get started.")
    col1, col2, col3 = st.columns(3)
    col1.markdown("### 1️⃣ Ingest\nUpload raw, messy data. The engine automatically handles missing values and formats.")
    col2.markdown("### 2️⃣ Train\nOur algorithms dynamically detect task types and train multiple models simultaneously.")
    col3.markdown("### 3️⃣ Predict\nDeploy the winning model instantly for single-record or bulk batch predictions.")

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
                
                trainer = ModelTrainer(X, y)
                results_df = trainer.train_and_evaluate()
                
                if not os.path.exists("outputs"):
                    os.makedirs("outputs")
                trainer.save_best_model("outputs")
                results_df.to_csv("outputs/performance_report.csv", index=False)
                
                st.success("✅ Pipeline executed successfully! Scroll down for results.")
            except Exception as e:
                st.error(f"An error occurred: {e}")

    st.divider()

    # --- STEP 3: DISPLAY RESULTS ---
    report_path = "outputs/performance_report.csv"
    model_path = "outputs/best_model.pkl"
    
    if os.path.exists(report_path) and os.path.exists(model_path):
        st.header("📊 Model Performance Report")
        report = pd.read_csv(report_path)
        model = joblib.load(model_path)
        
        is_classification = 'Accuracy' in report.columns
        best_model_name = report.iloc[0]['Model']
        
        # PRO FEATURE 1: DOWNLOAD MODEL BUTTON
        with open(model_path, "rb") as f:
            st.download_button(
                label="📥 Export Trained AI Brain (.pkl)",
                data=f,
                file_name=f"best_model_{best_model_name.replace(' ', '_')}.pkl",
                mime="application/octet-stream"
            )
            
        st.write("") # Quick spacer
        
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
            col2.metric(label="🎯 R2-Score (Accuracy)", value=f"{best_score:.4f}")
            col3.metric(label="📉 RMSE (Error)", value=f"{rmse_score:,.2f}", delta="- lower is better", delta_color="inverse")
            
            st.dataframe(report.style.highlight_max(subset=['R2-Score'], color='#2ecc71').highlight_min(subset=['RMSE (Error)'], color='#2ecc71'), use_container_width=True)
            
        st.divider()

        # PRO FEATURE 2: ADVANCED EVALUATION VISUALS
        st.header("📈 Advanced Evaluation Visuals")
        try:
            prep_eval = Preprocessor(raw_df, target_column=target_col)
            X_eval, y_eval = prep_eval.process()
            y_pred_eval = model.predict(X_eval)

            fig_eval, ax_eval = plt.subplots(figsize=(7, 5))
            
            if is_classification:
                cm = confusion_matrix(y_eval, y_pred_eval)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_eval, cbar=False)
                ax_eval.set_title("Confusion Matrix (Where the AI gets confused)", fontsize=14)
                ax_eval.set_xlabel("Predicted Label", fontsize=12)
                ax_eval.set_ylabel("True Label", fontsize=12)
            else:
                sns.scatterplot(x=y_eval, y=y_pred_eval, alpha=0.6, color='#3498db', edgecolor='k', ax=ax_eval)
                min_val = min(y_eval.min(), y_pred_eval.min())
                max_val = max(y_eval.max(), y_pred_eval.max())
                ax_eval.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Perfect Prediction Line")
                ax_eval.set_title("Actual vs. Predicted Values", fontsize=14)
                ax_eval.set_xlabel("Actual Data", fontsize=12)
                ax_eval.set_ylabel("AI Predictions", fontsize=12)
                ax_eval.legend()
                
            st.pyplot(fig_eval)
        except Exception as e:
            st.warning(f"⚠️ Could not generate advanced visuals for this dataset. Details: {e}")

        st.divider()

        # --- EXPLAINABILITY ---
        st.header("🧠 Explainable AI (Model Brain Scan)")
        st.markdown(f"**Why does the {best_model_name} make its decisions?** This chart reveals which columns carry the most mathematical weight.")
        
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
                st.info("ℹ️ The current winning algorithm is a complex mathematical 'Black Box' and does not easily reveal its internal feature weighting.")
        except Exception as e:
            st.warning("⚠️ Could not generate feature importance chart for this specific dataset configuration.")
        
        st.divider()

        # Setup classes for both prediction types
        is_categorical = raw_df[target_col].dtype == 'object' or raw_df[target_col].dtype.name == 'category'
        original_classes = sorted(raw_df[target_col].dropna().unique()) if is_categorical else None

        # --- SINGLE PREDICTION ---
        st.header("🔮 Single Record Prediction")
        st.markdown("Edit the table below to represent a brand-new scenario, then click predict.")
        
        template_df = raw_df.drop(columns=[target_col]).iloc[[0]].copy()
        edited_df = st.data_editor(template_df, hide_index=True)
        
        if st.button("🧬 Predict Single Record", type="primary"):
            with st.spinner("Analyzing new data..."):
                edited_df[target_col] = raw_df[target_col].iloc[0] 
                combined_df = pd.concat([raw_df, edited_df], ignore_index=True)
                
                prep_new = Preprocessor(combined_df, target_column=target_col)
                X_new, _ = prep_new.process()
                
                final_unseen_features = X_new.iloc[[-1]]
                raw_prediction = model.predict(final_unseen_features)[0]
                
                if is_classification:
                    final_output = str(original_classes[int(raw_prediction)]) if is_categorical else str(int(raw_prediction))
                    st.success(f"### 🚨 Final AI Diagnosis/Category: **{final_output}**")
                else:
                    st.success(f"### 📈 Final AI Predicted Value: **{raw_prediction:,.2f}**")

        st.divider()

        # --- PREMIUM FEATURE: BULK BATCH PREDICTION ---
        st.header("📦 Bulk Batch Prediction")
        st.markdown("Upload a new CSV file containing multiple rows (with the same columns as your training data) to predict them all at once.")
        
        batch_file = st.file_uploader("Upload Batch CSV for Predictions", type=["csv"], key="batch_data")
        
        if batch_file is not None:
            batch_df = pd.read_csv(batch_file)
            st.info(f"Loaded {len(batch_df)} rows for batch prediction.")
            
            if st.button("🚀 Run Batch Prediction", type="primary"):
                with st.spinner("Processing batch data..."):
                    try:
                        batch_process_df = batch_df.copy()
                        if target_col not in batch_process_df.columns:
                            batch_process_df[target_col] = raw_df[target_col].iloc[0] # Dummy value
                            
                        combined_batch_df = pd.concat([raw_df, batch_process_df], ignore_index=True)
                        
                        prep_batch = Preprocessor(combined_batch_df, target_column=target_col)
                        X_batch, _ = prep_batch.process()
                        
                        final_batch_features = X_batch.iloc[-len(batch_df):]
                        batch_predictions = model.predict(final_batch_features)
                        
                        if is_classification:
                            formatted_preds = [original_classes[int(p)] if is_categorical else int(p) for p in batch_predictions]
                        else:
                            formatted_preds = [round(p, 2) for p in batch_predictions]
                            
                        final_batch_df = batch_df.copy()
                        final_batch_df[f'AI_Predicted_{target_col}'] = formatted_preds
                        
                        st.success("✅ Batch Prediction Complete! See preview below:")
                        st.dataframe(final_batch_df.head(10), use_container_width=True)
                        
                        csv_data = final_batch_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Full Predictions as CSV",
                            data=csv_data,
                            file_name="ai_batch_predictions.csv",
                            mime="text/csv"
                        )
                    except Exception as e:
                        st.error(f"⚠️ Error during batch prediction. Make sure your uploaded file matches the original data structure! Details: {e}")