import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

class Preprocessor:
    def __init__(self, df, target_column):
        self.df = df.copy()
        self.target_column = target_column
        self.label_encoders = {}
        self.scaler = StandardScaler()

    def encode_categorical(self):
        cat_cols = self.df.select_dtypes(include=['object']).columns
        
        for col in cat_cols:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col].astype(str))
            self.label_encoders[col] = le
            
        print(f"[*] Encoded categorical columns: {list(cat_cols)}")
        return self

    def scale_numerical(self):
        # We don't want to scale the target column if it's numerical, so we exclude it
        num_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        features_to_scale = [col for col in num_cols if col != self.target_column]

        if len(features_to_scale) > 0:
            self.df[features_to_scale] = self.scaler.fit_transform(self.df[features_to_scale])
            print(f"[*] Scaled numerical features: {features_to_scale}")
            
        return self

    def process(self):
        print("--- Starting Data Preprocessing ---")
        
        # Check if target column exists
        if self.target_column not in self.df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in dataset!")

        self.encode_categorical()
        self.scale_numerical()
        
        # Separate Features (X) and Target (y)
        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]
        
        print("--- Data Preprocessing Complete ---\n")
        return X, y