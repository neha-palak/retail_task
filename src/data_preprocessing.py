import pandas as pd
import numpy as np
import os

def preprocess_data(file_path, target_col="avg_purchase_value"):
    df = pd.read_csv(file_path)

    numeric_df = df.select_dtypes(include=[np.number]).copy()

    if target_col not in numeric_df.columns:
        raise ValueError(f"Target column '{target_col}' must be numeric and present in dataset.")

    # correlations with target
    corrs = numeric_df.corr()[target_col].drop(target_col)

    # top 10 correlations
    top_features = corrs.abs().sort_values(ascending=False).head(10).index.tolist()

    # Prepare X and y
    X = numeric_df[top_features].values
    y = numeric_df[target_col].values

    return X, y, top_features

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    DATA_FILE = os.path.join(BASE_DIR, "data", "Retail Raw.csv")
    X, y, feats = preprocess_data(DATA_FILE)
    print("Preprocessing complete.")
    print("Top 10 features used:", feats)
    print("Feature shape:", X.shape, "Target shape:", y.shape)
