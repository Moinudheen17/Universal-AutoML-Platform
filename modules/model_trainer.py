import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
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


class ModelTrainer:
    def __init__(self, X, y):
        self.X    = X
        self.y    = y
        self.best_model      = None
        self.best_model_name = ""
        self.all_models      = {}
        self.cv_scores       = {}
        self.task_type       = self._determine_task_type()

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
                    n_estimators=100, eval_metric='logloss',
                    random_state=42, verbosity=0
                )
            if LGBM_AVAILABLE:
                models["LightGBM"] = LGBMClassifier(
                    n_estimators=100, random_state=42, verbose=-1
                )
            return models
        else:
            models = {
                "Linear Regression":        LinearRegression(),
                "Support Vector Regressor": SVR(),
                "Random Forest":            RandomForestRegressor(n_estimators=100, random_state=42),
            }
            if XGBOOST_AVAILABLE:
                models["XGBoost"] = XGBRegressor(
                    n_estimators=100, random_state=42, verbosity=0
                )
            if LGBM_AVAILABLE:
                models["LightGBM"] = LGBMRegressor(
                    n_estimators=100, random_state=42, verbose=-1
                )
            return models

    def train_and_evaluate(self):
        print(f"[*] Task Detected: {self.task_type}")

        # Convert y to clean numpy array
        # XGBoost/LightGBM require numeric integer labels — always encode
        self._y_encoder = LabelEncoder()
        if self.task_type == "Classification":
            y_encoded = self._y_encoder.fit_transform(self.y.astype(str))
        else:
            y_encoded = np.array(self.y, dtype=float)

        # Convert X to numpy with safe column names
        X_np = np.array(self.X, dtype=float)

        X_train, X_test, y_train, y_test = train_test_split(
            X_np, y_encoded, test_size=0.2, random_state=42
        )

        models     = self._get_models()
        results    = []
        best_score = -float('inf')
        cv_folds   = min(5, len(y_train) // 10) if len(y_train) > 50 else 3

        for name, model in models.items():
            print(f"[*] Training {name}...")
            try:
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                self.all_models[name] = model

                if self.task_type == "Classification":
                    acc = accuracy_score(y_test, predictions)
                    f1  = f1_score(y_test, predictions, average='weighted')
                    try:
                        cv      = cross_val_score(model, X_np, y_encoded, cv=cv_folds, scoring='accuracy')
                        cv_mean = float(cv.mean())
                        cv_std  = float(cv.std())
                    except Exception:
                        cv_mean, cv_std = acc, 0.0
                    self.cv_scores[name] = {"mean": cv_mean, "std": cv_std}
                    results.append({"Model": name, "Accuracy": acc, "F1-Score": f1, "CV Mean": cv_mean, "CV Std": cv_std})
                    if acc > best_score:
                        best_score           = acc
                        self.best_model      = model
                        self.best_model_name = name
                else:
                    r2   = r2_score(y_test, predictions)
                    rmse = np.sqrt(mean_squared_error(y_test, predictions))
                    try:
                        cv      = cross_val_score(model, X_np, y_encoded, cv=cv_folds, scoring='r2')
                        cv_mean = float(cv.mean())
                        cv_std  = float(cv.std())
                    except Exception:
                        cv_mean, cv_std = r2, 0.0
                    self.cv_scores[name] = {"mean": cv_mean, "std": cv_std}
                    results.append({"Model": name, "R2-Score": r2, "RMSE (Error)": rmse, "CV Mean": cv_mean, "CV Std": cv_std})
                    if r2 > best_score:
                        best_score           = r2
                        self.best_model      = model
                        self.best_model_name = name

            except Exception as e:
                print(f"[!] {name} failed: {e}")
                continue

        print(f"[🏆] Best Model: {self.best_model_name}")
        return pd.DataFrame(results).sort_values(
            by=list(results[0].keys())[1], ascending=False
        ).reset_index(drop=True)

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
