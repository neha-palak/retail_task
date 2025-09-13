import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_FILE = os.path.join(DATA_DIR, "Retail Raw.csv")
PROCESSED_FILE = os.path.join(DATA_DIR, "processed_data.csv")
FEATURES_FILE = os.path.join(DATA_DIR, "selected_features.txt")

os.makedirs(DATA_DIR, exist_ok=True)

df = pd.read_csv(RAW_FILE)

numeric_df = df.select_dtypes(include=[np.number]).copy()
target_col = "avg_purchase_value"
if target_col not in numeric_df.columns:
        raise ValueError(f"Target column '{target_col}' must be numeric and present in dataset.")

# correlations with target
corrs = numeric_df.corr()[target_col].drop(target_col)

# top 10 correlations
top_features = corrs.abs().sort_values(ascending=False).head(10).index.tolist()

df = df[top_features + [target_col]] 

df.to_csv(PROCESSED_FILE, index=False)

with open(FEATURES_FILE, "w") as f:
    f.write("Top 10 selected numeric features:\n")
    for feat in top_features:
        f.write(f"{feat}\n")

print(f"Processed data saved to {PROCESSED_FILE}")
print(f"Selected features saved to {FEATURES_FILE}")
print(f"Shape of processed data: {df.shape}")
print("Top 10 numeric features:", top_features)
