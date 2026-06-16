import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer


class Preprocessor:
    def __init__(self, df, target_column):
        self.df = df.copy()
        self.target_column = target_column
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.tfidf_vectorizers = {}
        self.text_columns = []
        self.categorical_columns = []

    # ── COLUMN TYPE DETECTION ────────────────────────────────────────────────

    def _detect_column_types(self):
        """
        Smartly separates object columns into:
        - TEXT columns: high cardinality, long strings → needs TF-IDF
        - CATEGORICAL columns: few unique values, short strings → LabelEncoder
        """
        object_cols = [
            col for col in self.df.select_dtypes(include=['object']).columns
            if col != self.target_column
        ]

        for col in object_cols:
            unique_ratio = self.df[col].nunique() / len(self.df)
            avg_length   = self.df[col].dropna().astype(str).str.len().mean()

            # Text column: high cardinality OR long average string length
            if unique_ratio > 0.3 or avg_length > 20:
                self.text_columns.append(col)
                print(f"[TEXT]  Detected '{col}' as text column "
                      f"(unique_ratio={unique_ratio:.2f}, avg_len={avg_length:.1f})")
            else:
                self.categorical_columns.append(col)
                print(f"[CAT]   Detected '{col}' as categorical column "
                      f"(unique_ratio={unique_ratio:.2f})")

        # Target column — if object, it's always categorical (label encode)
        if self.target_column in self.df.select_dtypes(include=['object']).columns:
            if self.target_column not in self.categorical_columns:
                self.categorical_columns.append(self.target_column)

    # ── ENCODING ─────────────────────────────────────────────────────────────

    def _encode_text_columns(self):
        """
        TF-IDF encode high-cardinality text columns.
        Converts each text column into multiple numeric TF-IDF feature columns.
        """
        for col in self.text_columns:
            print(f"[TF-IDF] Vectorizing '{col}'...")
            tfidf = TfidfVectorizer(max_features=50, stop_words='english')
            tfidf_matrix = tfidf.fit_transform(
                self.df[col].fillna('').astype(str)
            ).toarray()

            # Create new columns: text_col_word1, text_col_word2 ...
            tfidf_cols = [f"{col}_tfidf_{i}" for i in range(tfidf_matrix.shape[1])]
            tfidf_df   = pd.DataFrame(tfidf_matrix, columns=tfidf_cols, index=self.df.index)

            self.df = pd.concat([self.df.drop(columns=[col]), tfidf_df], axis=1)
            self.tfidf_vectorizers[col] = tfidf
            print(f"[TF-IDF] '{col}' → {tfidf_matrix.shape[1]} features")

    def _encode_categorical_columns(self):
        """
        LabelEncoder for low-cardinality categorical columns (including target).
        """
        for col in self.categorical_columns:
            if col not in self.df.columns:
                continue
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col].astype(str))
            self.label_encoders[col] = le
            print(f"[LABEL] Encoded '{col}' → {len(le.classes_)} classes")

    # ── SCALING ──────────────────────────────────────────────────────────────

    def scale_numerical(self):
        """Standard scale all numeric feature columns (not the target)."""
        num_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        features_to_scale = [col for col in num_cols if col != self.target_column]

        if features_to_scale:
            self.df[features_to_scale] = self.scaler.fit_transform(
                self.df[features_to_scale]
            )
            print(f"[SCALE] Scaled {len(features_to_scale)} numeric features")

        return self

    # ── TARGET ENCODER ───────────────────────────────────────────────────────

    def get_target_encoder(self):
        """
        Returns the LabelEncoder for the target column so app.py can call
        inverse_transform() to decode 0/1 back to Yes/No etc.
        Returns None if target was already numeric.
        """
        return self.label_encoders.get(self.target_column, None)

    def get_column_report(self):
        """Returns a summary dict of how each column was handled."""
        return {
            "text_columns":        self.text_columns,
            "categorical_columns": self.categorical_columns,
            "tfidf_features":      {col: f"{v.max_features} features"
                                    for col, v in self.tfidf_vectorizers.items()},
        }

    # ── MAIN PROCESS ─────────────────────────────────────────────────────────

    def process(self):
        print("--- Starting Data Preprocessing ---")

        if self.target_column not in self.df.columns:
            raise ValueError(
                f"Target column '{self.target_column}' not found in dataset!"
            )

        # Step 1: Detect column types
        self._detect_column_types()

        # Step 2: TF-IDF for text columns
        if self.text_columns:
            self._encode_text_columns()

        # Step 3: LabelEncode categorical columns + target
        self._encode_categorical_columns()

        # Step 4: Scale numeric features
        self.scale_numerical()

        # Step 5: Separate X and y
        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]

        print(f"--- Preprocessing Complete | X: {X.shape} | y: {y.shape} ---\n")
        return X, y
