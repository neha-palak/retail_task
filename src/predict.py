# src/train_models.py

import numpy as np
import pickle
import os
import pandas as pd

print("Columns in dataset:", df.columns.tolist())
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)
print("First 5 rows of target:", y[:5])


# ------------------------
# Utility functions
# ------------------------

def add_bias(X):
    return np.c_[np.ones((X.shape[0], 1)), X]

def linear_regression(X, y):
    X_b = add_bias(X)
    return np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y

def polynomial_features(X, degree=2):
    poly = X.copy()
    for d in range(2, degree + 1):
        poly = np.c_[poly, X ** d]
    return poly

def ridge_regression(X, y, lam=1.0):
    X_b = add_bias(X)
    n = X_b.shape[1]
    I = np.eye(n)
    I[0, 0] = 0  # do not regularize bias
    return np.linalg.pinv(X_b.T @ X_b + lam * I) @ X_b.T @ y

def lasso_regression(X, y, lam=0.1, max_iter=1000, tol=1e-6):
    X_b = add_bias(X)
    m, n = X_b.shape
    theta = np.zeros(n)
    
    for _ in range(max_iter):
        theta_old = theta.copy()
        for j in range(n):
            X_j = X_b[:, j]
            residual = y - X_b @ theta + theta[j] * X_j
            rho = X_j.T @ residual
            if j == 0:
                theta[j] = rho / (X_j.T @ X_j)
            else:
                if rho < -lam / 2:
                    theta[j] = (rho + lam / 2) / (X_j.T @ X_j)
                elif rho > lam / 2:
                    theta[j] = (rho - lam / 2) / (X_j.T @ X_j)
                else:
                    theta[j] = 0
        if np.linalg.norm(theta - theta_old, ord=1) < tol:
            break
    return theta

def evaluate(X, y, theta):
    X_b = add_bias(X)
    preds = X_b @ theta
    mse = np.mean((y - preds) ** 2)
    rmse = np.sqrt(mse)
    ss_res = np.sum((y - preds) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot
    return mse, rmse, r2

# ------------------------
# Load preprocessed data
# ------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_FILE = os.path.join(BASE_DIR, "data", "processed_data.csv")

df = pd.read_csv(PROCESSED_FILE)

# Choose target and features
target_col = "avg_purchase_value"  # or "churned" if predicting churn
y = df[target_col].values
X = df.drop(columns=[target_col]).values

# ------------------------
# Train models
# ------------------------
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

theta_lin = linear_regression(X, y)
with open(os.path.join(MODELS_DIR, "regression_model_lin.pkl"), "wb") as f:
    pickle.dump(theta_lin, f)

X_poly = polynomial_features(X, degree=2)
theta_poly = linear_regression(X_poly, y)
with open(os.path.join(MODELS_DIR, "regression_model_poly.pkl"), "wb") as f:
    pickle.dump((theta_poly, 2), f)

theta_ridge = ridge_regression(X, y, lam=10)
with open(os.path.join(MODELS_DIR, "regression_model_ridge.pkl"), "wb") as f:
    pickle.dump(theta_ridge, f)

theta_lasso = lasso_regression(X, y, lam=0.1)
with open(os.path.join(MODELS_DIR, "regression_model_lasso.pkl"), "wb") as f:
    pickle.dump(theta_lasso, f)

# ------------------------
# Evaluate models
# ------------------------
mse1, rmse1, r21 = evaluate(X, y, theta_lin)
mse2, rmse2, r22 = evaluate(X_poly, y, theta_poly)
mse3, rmse3, r23 = evaluate(X, y, theta_ridge)
mse4, rmse4, r24 = evaluate(X, y, theta_lasso)

print("\nModel Evaluation Results:")
print(f"Linear Regression     -> R² = {r21:.4f}, RMSE = {rmse1:.4f}")
print(f"Polynomial Regression -> R² = {r22:.4f}, RMSE = {rmse2:.4f}")
print(f"Ridge Regression      -> R² = {r23:.4f}, RMSE = {rmse3:.4f}")
print(f"Lasso Regression      -> R² = {r24:.4f}, RMSE = {rmse4:.4f}")

# Pick best model (based on R²)
best = max([(r21, "lin", theta_lin),
            (r22, "poly", (theta_poly, 2)),
            (r23, "ridge", theta_ridge),
            (r24, "lasso", theta_lasso)], key=lambda x: x[0])

with open(os.path.join(MODELS_DIR, "regression_model_final.pkl"), "wb") as f:
    pickle.dump(best[2], f)

print(f"\nBest Model: {best[1]} with R² = {best[0]:.4f}")
