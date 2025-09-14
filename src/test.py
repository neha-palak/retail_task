import pandas as pd 
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_FILE = os.path.join(DATA_DIR, "Retail.csv")  # rename raw data as such
PROCESSED_FILE = os.path.join(DATA_DIR, "processed_data.csv")
FEATURES_FILE = os.path.join(DATA_DIR, "selected_features.txt")

os.makedirs(DATA_DIR, exist_ok=True)

def load_data(path):
    return pd.read_csv(path)

def preprocess_data(path, target_col="avg_purchase_value", save_csv=True):
    df = load_data(path)
    print(f"Loaded dataset from {path}")

    # clean column names
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
        .str.replace("/", "_")
        .str.replace("(", "")
        .str.replace(")", "")
    )

    # drop duplicates
    df = df.drop_duplicates()

    # drop large cardinality object columns (likely IDs or dates)
    thresh = 1000
    object_cols = df.select_dtypes(include="object").columns
    large = [col for col in object_cols if df[col].nunique() > thresh]
    if large:
        print("Dropping large cardinality columns:\n", large)
        df = df.drop(columns=large)

    # drop obvious ID-like columns
    drop_cols = [
        "product_id", "store_zip_code", "promotion_id",
        "customer_zip_code", "transaction_hour", "transaction_id"
    ]
    df = df.drop([col for col in drop_cols if col in df.columns], axis=1, errors="ignore")

    # fill missing values
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].mean())

    # correlation before encoding
    corr = df.corr(numeric_only=True)[target_col].drop(target_col)

    # top 30 positive and negative features
    top_features = corr.sort_values(ascending=False).head(30).index.tolist()
    bottom_features = corr.sort_values(ascending=True).head(30).index.tolist()

    print("Top 30 positively correlated features with target:")
    print(top_features)
    print("\nTop 30 negatively correlated features with target:")
    print(bottom_features)

    # encode categorical variables
    object_cols = df.select_dtypes(include="object").columns.tolist()
    if object_cols:
        print("Encoding categorical columns:", object_cols)
        df = pd.get_dummies(df, columns=object_cols, drop_first=True).astype(int)

    # standardize numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()

    # keep only selected features
    selected_columns = [target_col] + top_features + bottom_features
    df = df[[col for col in selected_columns if col in df.columns]]

    # save processed data
    if save_csv:
        df.to_csv(PROCESSED_FILE, index=False)

    # save selected features list
    with open(FEATURES_FILE, "w") as f:
        f.write("30 Positively correlated features:\n")
        for feat in top_features:
            f.write(f"{feat}\n")

        f.write("\n30 Negatively correlated features:\n")
        for feat in bottom_features:
            f.write(f"{feat}\n")

    print(f"Processed data saved to {PROCESSED_FILE}")
    print(f"Selected features saved to {FEATURES_FILE}")
    print(f"Shape of processed data: {df.shape}")

    return df

if __name__ == "__main__":
    preprocess_data(RAW_FILE)
