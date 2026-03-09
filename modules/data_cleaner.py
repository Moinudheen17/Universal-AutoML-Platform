import pandas as pd
from sklearn.impute import SimpleImputer
import warnings

# Ignore minor warnings to keep the terminal output clean
warnings.filterwarnings('ignore')

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()

    def remove_duplicates(self):
        initial_shape = self.df.shape
        self.df = self.df.drop_duplicates()
        final_shape = self.df.shape
        print(f"[*] Dropped {initial_shape[0] - final_shape[0]} duplicate rows.")
        return self

    def handle_missing_values(self):
        # Identify numerical and categorical columns
        num_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        cat_cols = self.df.select_dtypes(include=['object']).columns

        # Fill numerical missing values with the median
        if len(num_cols) > 0:
            num_imputer = SimpleImputer(strategy='median')
            self.df[num_cols] = num_imputer.fit_transform(self.df[num_cols])
            print(f"[*] Filled missing numerical values in: {list(num_cols)}")

        # Fill categorical missing values with the most frequent value (mode)
        if len(cat_cols) > 0:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            self.df[cat_cols] = cat_imputer.fit_transform(self.df[cat_cols])
            print(f"[*] Filled missing categorical values in: {list(cat_cols)}")

        return self.df

    def clean(self):
        print("\n--- Starting Data Cleaning ---")
        self.remove_duplicates()
        cleaned_df = self.handle_missing_values()
        print("--- Data Cleaning Complete ---\n")
        return cleaned_df