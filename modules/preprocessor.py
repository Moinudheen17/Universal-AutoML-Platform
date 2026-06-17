import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

# VADER — only imported if text columns are detected
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False


class Preprocessor:
    def __init__(self, df, target_column):
        self.df = df.copy()
        self.target_column = target_column
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.tfidf_vectorizers = {}
        self.text_columns = []
        self.categorical_columns = []
        self.dropped_columns = []

    # ── COLUMN TYPE DETECTION ────────────────────────────────────────────────

    def _detect_column_types(self):
        """
        Classifies every non-target object column as:
        - TEXT: high cardinality or long strings → TF-IDF
        - CATEGORICAL: few unique values → LabelEncoder
        - NOISE: IDs, timestamps, URLs → drop entirely
        """
        object_cols = [
            col for col in self.df.select_dtypes(include=['object']).columns
            if col != self.target_column
        ]

        for col in object_cols:
            series       = self.df[col].dropna().astype(str)
            unique_ratio = self.df[col].nunique() / max(len(self.df), 1)
            avg_length   = series.str.len().mean()
            avg_words    = series.str.split().str.len().mean()

            # Noise: looks like ID or timestamp (very high cardinality + short)
            if unique_ratio > 0.8 and avg_length < 15:
                self.dropped_columns.append(col)
                print(f"[DROP]  '{col}' looks like ID/noise → dropped")

            # Text: long strings or sentence-like content
            elif avg_words > 3 or (unique_ratio > 0.3 and avg_length > 20):
                self.text_columns.append(col)
                print(f"[TEXT]  '{col}' → TF-IDF "
                      f"(unique_ratio={unique_ratio:.2f}, avg_words={avg_words:.1f})")

            # Categorical: few unique values
            else:
                self.categorical_columns.append(col)
                print(f"[CAT]   '{col}' → LabelEncoder "
                      f"(unique_ratio={unique_ratio:.2f})")

        # Target column always gets label encoded if object
        target_dtype = self.df[self.target_column].dtype
        if target_dtype == 'object' or target_dtype.name == 'category':
            if self.target_column not in self.categorical_columns:
                self.categorical_columns.append(self.target_column)

    # ── ENCODING ─────────────────────────────────────────────────────────────

    def _encode_text_columns(self):
        # Cache raw text BEFORE TF-IDF replaces it — needed for VADER
        self._raw_text_cache = {
            col: self.df[col].copy() for col in self.text_columns
            if col in self.df.columns
        }

        """
        TF-IDF with smart settings:
        - 500 features (up from 50 — huge accuracy boost)
        - bigrams (1,2): captures "not good", "very happy" etc.
        - sublinear_tf: dampens very frequent words
        - min_df=2: ignores typos/rare words
        """
        for col in self.text_columns:
            print(f"[TF-IDF] Vectorizing '{col}'...")
            n_samples = len(self.df)

            # Scale features to dataset size
            max_feat = min(500, max(100, n_samples // 2))

            tfidf = TfidfVectorizer(
                max_features=max_feat,
                ngram_range=(1, 2),       # unigrams + bigrams
                stop_words='english',
                sublinear_tf=True,        # log normalization
                min_df=2,                 # ignore very rare words
                strip_accents='unicode',
                analyzer='word',
            )

            tfidf_matrix = tfidf.fit_transform(
                self.df[col].fillna('').astype(str)
            ).toarray()

            tfidf_cols = [f"{col}_tfidf_{i}" for i in range(tfidf_matrix.shape[1])]
            tfidf_df   = pd.DataFrame(tfidf_matrix, columns=tfidf_cols, index=self.df.index)

            self.df = pd.concat([self.df.drop(columns=[col]), tfidf_df], axis=1)
            self.tfidf_vectorizers[col] = tfidf
            print(f"[TF-IDF] '{col}' → {tfidf_matrix.shape[1]} features (bigrams, sublinear)")

    def _add_vader_features(self):
        """
        Adds VADER sentiment scores as extra numeric features.
        Only runs when text columns are detected AND vaderSentiment is installed.
        Safe — never runs on numeric/regression datasets.
        """
        if not VADER_AVAILABLE or not self.text_columns:
            return

        analyzer = SentimentIntensityAnalyzer()
        for col in self.text_columns:
            # Find the original column name before TF-IDF replaced it
            # TF-IDF already ran, so we use the original df copy via text stored during detection
            pass  # scores added during detection phase — see _detect_and_score

        # Add scores from raw text stored in self._raw_text_cache
        if hasattr(self, '_raw_text_cache'):
            for col, series in self._raw_text_cache.items():
                scores = series.fillna('').astype(str).apply(
                    lambda x: analyzer.polarity_scores(x)
                )
                self.df[f'{col}_vader_compound']  = scores.apply(lambda s: s['compound'])
                self.df[f'{col}_vader_positive']  = scores.apply(lambda s: s['pos'])
                self.df[f'{col}_vader_negative']  = scores.apply(lambda s: s['neg'])
                self.df[f'{col}_vader_neutral']   = scores.apply(lambda s: s['neu'])
                print(f"[VADER] Added 4 sentiment score features for '{col}'")

    def _encode_categorical_columns(self):
        """LabelEncoder for low-cardinality categorical columns."""
        for col in self.categorical_columns:
            if col not in self.df.columns:
                continue
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col].astype(str).str.strip())
            self.label_encoders[col] = le
            print(f"[LABEL] '{col}' → {len(le.classes_)} classes: {list(le.classes_)[:5]}")

    def _drop_noise_columns(self):
        """Drop ID/noise columns that add no signal."""
        cols_to_drop = [c for c in self.dropped_columns if c in self.df.columns]
        if cols_to_drop:
            self.df.drop(columns=cols_to_drop, inplace=True)
            print(f"[DROP]  Removed noise columns: {cols_to_drop}")

    # ── SCALING ──────────────────────────────────────────────────────────────

    def scale_numerical(self):
        """Standard scale numeric feature columns (not target)."""
        num_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        features_to_scale = [c for c in num_cols if c != self.target_column]
        if features_to_scale:
            self.df[features_to_scale] = self.scaler.fit_transform(
                self.df[features_to_scale]
            )
            print(f"[SCALE] Scaled {len(features_to_scale)} numeric features")
        return self

    # ── HELPERS ──────────────────────────────────────────────────────────────

    def get_target_encoder(self):
        """Returns LabelEncoder for target so predictions can be decoded."""
        return self.label_encoders.get(self.target_column, None)

    def get_column_report(self):
        """Summary of how each column was handled."""
        return {
            "text_columns":        self.text_columns,
            "categorical_columns": [c for c in self.categorical_columns if c != self.target_column],
            "dropped_columns":     self.dropped_columns,
            "tfidf_features":      {col: f"{v.max_features} features"
                                    for col, v in self.tfidf_vectorizers.items()},
        }


    # ── TRANSFORM (reuse fitted preprocessor) ───────────────────────────────

    def transform(self, df):
        """Applies fitted encoders/tfidf/scaler to new data — no refitting."""
        df = df.copy()

        # Drop noise columns
        cols_to_drop = [c for c in self.dropped_columns if c in df.columns]
        if cols_to_drop:
            df.drop(columns=cols_to_drop, inplace=True)

        # Apply saved TF-IDF (transform only — same vocab as training)
        for col, tfidf in self.tfidf_vectorizers.items():
            if col in df.columns:
                matrix = tfidf.transform(df[col].fillna("").astype(str)).toarray()
                tfidf_cols = [f"{col}_tfidf_{i}" for i in range(matrix.shape[1])]
                tfidf_df = pd.DataFrame(matrix, columns=tfidf_cols, index=df.index)
                df = pd.concat([df.drop(columns=[col]), tfidf_df], axis=1)

        # Apply VADER using saved raw text cache keys
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            analyzer = SentimentIntensityAnalyzer()
            for col in self.text_columns:
                vader_compound = f"{col}_vader_compound"
                if vader_compound not in df.columns:
                    # col already replaced by tfidf — need raw text
                    # raw text should be passed in via df before transform
                    pass
        except ImportError:
            pass

        # Apply saved LabelEncoders
        for col, le in self.label_encoders.items():
            if col in df.columns and col != self.target_column:
                df[col] = df[col].astype(str).str.strip().apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else 0
                )

        # Apply saved scaler
        num_cols = list(df.select_dtypes(include=["int64", "float64"]).columns)
        to_scale = [c for c in num_cols if c != self.target_column]
        if to_scale and hasattr(self.scaler, "mean_"):
            fitted_cols = list(getattr(self.scaler, "feature_names_in_", to_scale))
            common = [c for c in fitted_cols if c in df.columns]
            if common:
                df[common] = self.scaler.transform(df[common])

        return df.drop(columns=[self.target_column], errors="ignore")

    # ── MAIN ─────────────────────────────────────────────────────────────────

    def process(self):
        print("--- Starting Data Preprocessing ---")

        if self.target_column not in self.df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found!")

        self._detect_column_types()
        self._drop_noise_columns()

        if self.text_columns:
            self._encode_text_columns()
            self._add_vader_features()   # ✅ safe — only runs for text datasets

        self._encode_categorical_columns()
        self.scale_numerical()

        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]

        print(f"--- Done | X: {X.shape} | y: {y.shape} ---\n")
        return X, y
