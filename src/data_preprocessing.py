# src/data_preprocessing.py

import os
import pandas as pd
import numpy as np

# ------------------------
# Paths
# ------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_FILE = os.path.join(DATA_DIR, "Retail Raw.csv")  # match your filename
PROCESSED_FILE = os.path.join(DATA_DIR, "processed_data.csv")

os.makedirs(DATA_DIR, exist_ok=True)

# ------------------------
# Load raw data
# ------------------------
df = pd.read_csv(RAW_FILE)

# ------------------------
# Encode target
# ------------------------
binary_map = {"Yes": 1, "No": 0}
if "churned" in df.columns:
    df["churned"] = df["churned"].map(binary_map)

# ------------------------
# Encode binary columns
# ------------------------
binary_cols = ["loyalty_program", "weekend", "email_subscriptions"]
for col in binary_cols:
    if col in df.columns:
        df[col] = df[col].map(binary_map)

# ------------------------
# Encode ordinal columns
# ------------------------
ordinal_cols = {"income_bracket": ["Low", "Medium", "High"]}

for col, order in ordinal_cols.items():
    mapping = {val: idx for idx, val in enumerate(order)}
    if col in df.columns:
        df[col] = df[col].map(mapping)

# ------------------------
# One-hot encode nominal categorical columns
# ------------------------
nominal_cols = ["gender", "marital_status", "education_level", "occupation",
                "season", "app_usage", "social_media_engagement"]

for col in nominal_cols:
    if col in df.columns:
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
        df = pd.concat([df.drop(col, axis=1), dummies], axis=1)

# ------------------------
# Convert datetime columns to numeric (seconds since epoch)
# ------------------------
for col in df.columns:
    if df[col].dtype == "object":
        try:
            df[col] = pd.to_datetime(df[col])
            df[col] = df[col].astype(np.int64) // 10**9
        except:
            # Not datetime, drop column
            df.drop(columns=[col], inplace=True)

# ------------------------
# Ensure all remaining columns are numeric
# ------------------------
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
df = df[numeric_cols]

# ------------------------
# Save processed CSV
# ------------------------
df.to_csv(PROCESSED_FILE, index=False)
print(f"Processed data saved to {PROCESSED_FILE}")
print(f"Shape of processed data: {df.shape}")
