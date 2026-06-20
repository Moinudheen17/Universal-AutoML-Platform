import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False


class ModelTrainer:
    def __init__(self, X, y, enable_tuning=True, enable_smote=True):
        self.X = X
        self.y = y
        self.enable_tuning = enable_tuning
        self.enable_smote  = enable_smote
        self.best_model      = None
        self.best_model_name = ""
        self.all_models      = {}
        self.cv_scores       = {}
        self.tuning_log       = {}   # before/after tuning scores
        self.smote_applied    = False
        self.task_type        = self._determine_task_type()

    def _determine_task_type(self):
        if self.y.dtype == 'object' or self.y.dtype.name == 'category':
            return "Classification"
        unique_values = self.y.nunique()
        unique_ratio  = unique_values / len(self.y)
        if unique_values <= 10 and unique_ratio < 0.05:
            return "Classification"
        return "Regression"

    def _get_models(self):
        if self.task_type == "Classification":
            models = {
                "Logistic Regression":       LogisticRegression(max_iter=1000),
                "Support Vector Classifier": SVC(probability=True),
                "Random Forest":             RandomForestClassifier(n_estimators=100, random_state=42),
            }
            if XGBOOST_AVAILABLE:
                models["XGBoost"] = XGBClassifier(
                    n_estimators=100, eval_metric='logloss', random_state=42, verbosity=0
                )
            if LGBM_AVAILABLE:
                models["LightGBM"] = LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
            return models
        else:
            models = {
                "Linear Regression":        LinearRegression(),
                "Support Vector Regressor": SVR(),
                "Random Forest":            RandomForestRegressor(n_estimators=100, random_state=42),
            }
            if XGBOOST_AVAILABLE:
                models["XGBoost"] = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
            if LGBM_AVAILABLE:
                models["LightGBM"] = LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
            return models

    def _get_param_grid(self, name):
        """Small, fast hyperparameter grids — keeps tuning quick for Streamlit Cloud."""
        grids = {
            "Random Forest": {
                "n_estimators": [100, 200, 300],
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
            },
            "XGBoost": {
                "n_estimators": [100, 200, 300],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
            },
            "LightGBM": {
                "n_estimators": [100, 200, 300],
                "max_depth": [-1, 5, 10],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
            },
            "Logistic Regression": {
                "C": [0.01, 0.1, 1, 5, 10],
            },
            "Support Vector Classifier": {
                "C": [0.1, 1, 5, 10],
                "kernel": ["linear", "rbf"],
            },
            "Support Vector Regressor": {
                "C": [0.1, 1, 5, 10],
                "kernel": ["linear", "rbf"],
            },
            "Linear Regression": {},  # no hyperparams worth tuning
        }
        return grids.get(name, {})

    def _apply_smote(self, X_train, y_train):
        """Balances classes if imbalance is significant. Classification only."""
        if not (self.enable_smote and SMOTE_AVAILABLE and self.task_type == "Classification"):
            return X_train, y_train

        unique, counts = np.unique(y_train, return_counts=True)
        if len(unique) < 2:
            return X_train, y_train

        imbalance_ratio = counts.min() / counts.max()
        if imbalance_ratio < 0.5:  # meaningful imbalance
            try:
                min_class_size = counts.min()
                k_neighbors    = min(5, max(1, min_class_size - 1))
                smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
                X_res, y_res = smote.fit_resample(X_train, y_train)
                self.smote_applied = True
                print(f"[SMOTE] Balanced classes: {dict(zip(unique, counts))} → resampled")
                return X_res, y_res
            except Exception as e:
                print(f"[SMOTE] Skipped: {e}")
        return X_train, y_train

    def train_and_evaluate(self):
        print(f"[*] Task Detected: {self.task_type}")

        self._y_encoder = LabelEncoder()
        if self.task_type == "Classification":
            y_encoded = self._y_encoder.fit_transform(self.y.astype(str))
        else:
            y_encoded = np.array(self.y, dtype=float)

        X_np = np.array(self.X, dtype=float)

        X_train, X_test, y_train, y_test = train_test_split(
            X_np, y_encoded, test_size=0.2, random_state=42
        )

        # Apply SMOTE only to training data (never test data)
        X_train_bal, y_train_bal = self._apply_smote(X_train, y_train)

        models     = self._get_models()
        results    = []
        best_score = -float('inf')
        cv_folds   = min(5, len(y_train) // 10) if len(y_train) > 50 else 3

        for name, model in models.items():
            print(f"[*] Training {name}...")
            try:
                # Baseline fit
                model.fit(X_train_bal, y_train_bal)
                baseline_preds = model.predict(X_test)
                baseline_score = (
                    accuracy_score(y_test, baseline_preds) if self.task_type == "Classification"
                    else r2_score(y_test, baseline_preds)
                )

                # Hyperparameter tuning (skip for Linear Regression / huge datasets)
                final_model = model
                tuned = False
                param_grid = self._get_param_grid(name)
                if self.enable_tuning and param_grid and len(X_train_bal) < 20000:
                    try:
                        search = RandomizedSearchCV(
                            model, param_grid,
                            n_iter=6, cv=min(3, cv_folds),
                            scoring='accuracy' if self.task_type == "Classification" else 'r2',
                            random_state=42, n_jobs=-1
                        )
                        search.fit(X_train_bal, y_train_bal)
                        final_model = search.best_estimator_
                        tuned = True
                    except Exception as e:
                        print(f"[!] Tuning skipped for {name}: {e}")

                predictions = final_model.predict(X_test)
                self.all_models[name] = final_model

                if self.task_type == "Classification":
                    acc = accuracy_score(y_test, predictions)
                    f1  = f1_score(y_test, predictions, average='weighted')
                    self.tuning_log[name] = {"before": baseline_score, "after": acc, "tuned": tuned}
                    try:
                        cv = cross_val_score(final_model, X_np, y_encoded, cv=cv_folds, scoring='accuracy')
                        cv_mean, cv_std = float(cv.mean()), float(cv.std())
                    except Exception:
                        cv_mean, cv_std = acc, 0.0
                    self.cv_scores[name] = {"mean": cv_mean, "std": cv_std}
                    results.append({"Model": name, "Accuracy": acc, "F1-Score": f1,
                                     "CV Mean": cv_mean, "CV Std": cv_std, "Tuned": tuned})
                    if acc > best_score:
                        best_score, self.best_model, self.best_model_name = acc, final_model, name
                else:
                    r2   = r2_score(y_test, predictions)
                    rmse = np.sqrt(mean_squared_error(y_test, predictions))
                    self.tuning_log[name] = {"before": baseline_score, "after": r2, "tuned": tuned}
                    try:
                        cv = cross_val_score(final_model, X_np, y_encoded, cv=cv_folds, scoring='r2')
                        cv_mean, cv_std = float(cv.mean()), float(cv.std())
                    except Exception:
                        cv_mean, cv_std = r2, 0.0
                    self.cv_scores[name] = {"mean": cv_mean, "std": cv_std}
                    results.append({"Model": name, "R2-Score": r2, "RMSE (Error)": rmse,
                                     "CV Mean": cv_mean, "CV Std": cv_std, "Tuned": tuned})
                    if r2 > best_score:
                        best_score, self.best_model, self.best_model_name = r2, final_model, name

            except Exception as e:
                print(f"[!] {name} failed: {e}")
                continue

        print(f"[🏆] Best Model: {self.best_model_name}")
        return pd.DataFrame(results).sort_values(
            by=list(results[0].keys())[1], ascending=False
        ).reset_index(drop=True)

    def get_tuning_summary(self):
        """Returns before/after tuning improvement per model."""
        return self.tuning_log

    def save_best_model(self, output_dir="outputs"):
        if self.best_model is not None:
            joblib.dump(self.best_model, f"{output_dir}/best_model.pkl")
            print(f"[*] Saved best model: {self.best_model_name}")

    def save_all_models(self, output_dir="outputs"):
        os.makedirs(output_dir, exist_ok=True)
        for name, model in self.all_models.items():
            safe_name = name.replace(" ", "_")
            joblib.dump(model, f"{output_dir}/model_{safe_name}.pkl")
            print(f"[*] Saved: {safe_name}")
