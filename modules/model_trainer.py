import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split

# Classification Models & Metrics
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# Regression Models & Metrics
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error


class ModelTrainer:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.best_model = None
        self.best_model_name = ""
        self.all_models = {}       # ✅ stores ALL trained models by name
        self.task_type = self._determine_task_type()

    def _determine_task_type(self):
        if self.y.dtype == 'object' or self.y.dtype.name == 'category':
            return "Classification"

        unique_values = self.y.nunique()
        total_rows    = len(self.y)
        unique_ratio  = unique_values / total_rows

        if unique_values <= 10 and unique_ratio < 0.05:
            return "Classification"

        return "Regression"

    def _get_models(self):
        if self.task_type == "Classification":
            return {
                "Logistic Regression":        LogisticRegression(max_iter=1000),
                "Support Vector Classifier":  SVC(probability=True),
                "Random Forest Classifier":   RandomForestClassifier()
            }
        else:
            return {
                "Linear Regression":        LinearRegression(),
                "Support Vector Regressor": SVR(),
                "Random Forest Regressor":  RandomForestRegressor()
            }

    def train_and_evaluate(self):
        print(f"[*] Task Detected: {self.task_type}")
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

        models     = self._get_models()
        results    = []
        best_score = -float('inf')

        for name, model in models.items():
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            # ✅ Save every trained model
            self.all_models[name] = model

            if self.task_type == "Classification":
                acc = accuracy_score(y_test, predictions)
                f1  = f1_score(y_test, predictions, average='weighted')
                results.append({"Model": name, "Accuracy": acc, "F1-Score": f1})
                if acc > best_score:
                    best_score = acc
                    self.best_model = model
                    self.best_model_name = name
            else:
                r2   = r2_score(y_test, predictions)
                rmse = np.sqrt(mean_squared_error(y_test, predictions))
                results.append({"Model": name, "R2-Score": r2, "RMSE (Error)": rmse})
                if r2 > best_score:
                    best_score = r2
                    self.best_model = model
                    self.best_model_name = name

        print(f"[🏆] Best Model: {self.best_model_name}")
        return pd.DataFrame(results).sort_values(
            by=list(results[0].keys())[1], ascending=False
        )

    def save_best_model(self, output_dir="outputs"):
        """Saves the auto-selected best model."""
        if self.best_model is not None:
            joblib.dump(self.best_model, f"{output_dir}/best_model.pkl")
            print(f"[*] Best model ({self.best_model_name}) saved.")

    def save_all_models(self, output_dir="outputs"):
        """✅ Saves ALL trained models so user can switch between them."""
        os.makedirs(output_dir, exist_ok=True)
        for name, model in self.all_models.items():
            safe_name = name.replace(" ", "_")
            joblib.dump(model, f"{output_dir}/model_{safe_name}.pkl")
            print(f"[*] Saved: {safe_name}")
