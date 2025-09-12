# src/data_preprocessing.py

import os
import pandas as pd

# ------------------------
# Paths
# ------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_FILE = os.path.join(DATA_DIR, "Retail Raw.csv")   # change name if your file is different
PROCESSED_FILE = os.path.join(DATA_DIR, "processed_data.csv")

# Create data folder if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)

# ------------------------
# Load data
# ------------------------
df = pd.read_csv(RAW_FILE)

# ------------------------
# Target encoding
# ------------------------
binary_map = {"Yes": 1, "No": 0}
df["churned"] = df["churned"].map(binary_map)

# ------------------------
# Binary columns encoding
# ------------------------
binary_cols = ["loyalty_program", "weekend", "email_subscriptions"]
for col in binary_cols:
    if col in df.columns:
        df[col] = df[col].map(binary_map)

# ------------------------
# Ordinal encoding manually
# ------------------------
ordinal_cols = {"income_bracket": ["Low", "Medium", "High"]}

for col, order in ordinal_cols.items():
    mapping = {val: idx for idx, val in enumerate(order)}
    if col in df.columns:
        df[col] = df[col].map(mapping)

# ------------------------
# One-hot encoding manually
# ------------------------
onehot_cols = ["gender", "marital_status", "education_level", "occupation", 
               "season", "app_usage", "social_media_engagement"]

for col in onehot_cols:
    if col in df.columns:
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
        df = pd.concat([df.drop(col, axis=1), dummies], axis=1)

# ------------------------
# Save processed file
# ------------------------
df.to_csv(PROCESSED_FILE, index=False)
print(f"Processed data saved to {PROCESSED_FILE}")
