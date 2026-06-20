# ⚙️ Universal AutoML Engine

> An end-to-end automated machine learning platform that takes any CSV — tabular, text, or mixed — and automatically detects the task, cleans the data, trains multiple models, tunes them, and serves predictions through a polished web UI.

**🔗 Live Demo:** [universal-automl-platform-yl5og9imjxyesvxbmnhlcm.streamlit.app](https://universal-automl-platform-yl5og9imjxyesvxbmnhlcm.streamlit.app/)

---

## 📸 Screenshots

| Landing Page | Model Results |
|---|---|
| ![Landing](docs/screenshots/landing.png) | ![Results](docs/screenshots/results.png) |

| Single Prediction | Batch Prediction |
|---|---|
| ![Predict](docs/screenshots/predict.png) | ![Batch](docs/screenshots/batch.png) |

*(Add your own screenshots to `docs/screenshots/` and update the paths above)*

---

## ✨ Features

### 🧠 Intelligent Pipeline
- **Auto task detection** — automatically identifies Classification vs Regression based on target column characteristics
- **Smart column typing** — distinguishes numeric, categorical, text, and noise/ID columns automatically
- **Text support** — TF-IDF vectorization (unigrams + bigrams) for free-text columns, with VADER sentiment scores added as extra signal
- **Data cleaning** — handles missing values, encodes categoricals, scales numeric features

### 🤖 Model Training
- Trains **5 models** in parallel: Logistic/Linear Regression, SVM, Random Forest, **XGBoost**, **LightGBM**
- **Hyperparameter tuning** via `RandomizedSearchCV` — auto-tunes the top models, shows before/after improvement
- **5-fold cross-validation** on every model for reliable, non-lucky accuracy scores
- **SMOTE class balancing** — automatically detects and corrects imbalanced classification datasets
- **Model switcher** — compare and select any trained model, not just the auto-picked best one

### 📊 Explainability & Reporting
- Confusion matrix / Actual-vs-Predicted plots
- Feature importance charts
- **SHAP summary plots** — industry-standard model explainability
- **Data Quality Score** — missing values, duplicates, skew, and class imbalance at a glance
- **One-click PDF report export** — model comparison, charts, and metadata in a polished document

### 🔮 Predictions
- Single-record prediction with editable input table and live confidence scores
- Bulk batch prediction via CSV upload with downloadable results
- Correct label decoding (`Yes`/`No` instead of raw `0`/`1`)

### 🎨 UX
- Fluid dark glassmorphism UI with a cyan/teal/blue color scheme
- Built-in sample datasets (Titanic, Iris, Boston Housing) — try the app with zero setup
- Tabbed layout: Results · Visuals & Explainability · Single Predict · Batch Predict

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| **Frontend** | Streamlit, custom CSS (glassmorphism) |
| **ML Models** | scikit-learn, XGBoost, LightGBM |
| **NLP** | TF-IDF (scikit-learn), VADER Sentiment |
| **Explainability** | SHAP |
| **Imbalance Handling** | imbalanced-learn (SMOTE) |
| **Reporting** | ReportLab (PDF generation) |
| **Visualization** | Matplotlib, Seaborn |
| **Deployment** | Streamlit Community Cloud |

---

## 🚀 Run Locally

```bash
git clone https://github.com/moinudheen17/universal-automl-platform.git
cd universal-automl-platform
pip install -r requirements.txt
streamlit run app.py
```

---

## 📂 Project Structure

```
universal-automl-platform/
├── app.py                      # Main Streamlit app
├── modules/
│   ├── data_cleaner.py         # Missing value handling
│   ├── preprocessor.py         # Encoding, scaling, TF-IDF, VADER
│   ├── model_trainer.py        # Training, tuning, SMOTE, CV
│   └── pdf_report.py           # PDF report generation
├── requirements.txt
└── README.md
```

---

## 🎯 What This Project Demonstrates

- End-to-end ML pipeline design (ingestion → cleaning → training → serving)
- Handling heterogeneous data (numeric, categorical, free text) in one unified system
- Production ML practices: cross-validation, hyperparameter tuning, class imbalance correction
- Model explainability (SHAP, feature importance) for trust and transparency
- Full-stack thinking — not just a notebook, but a deployed, usable product

---

## 👤 Author

**Moinudheen M A ** — MCA (AI & ML), Yenepoya Institute of Arts, Science, Commerce and Management
[GitHub](https://github.com/moinudheen17) · Built as part of MCA thesis project

---

## 📄 License

This project is open source and available for educational use.
