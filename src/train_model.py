import pandas as pd
import numpy as np
import pickle
import os

# ----------------------------
# Regression Functions
# ----------------------------
def add_bias(X):
    return np.c_[np.ones((X.shape[0], 1)), X]

def linear_regression(X, y):
    X_b = add_bias(X)
    theta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y
    return theta

def polynomial_features(X, degree=2):
    poly = X.copy()
    for d in range(2, degree + 1):
        poly = np.c_[poly, X ** d]
    return poly

def ridge_regression(X, y, lam=1.0):
    X_b = add_bias(X)
    n = X_b.shape[1]
    I = np.eye(n)
    I[0, 0] = 0  # don't regularize bias
    theta = np.linalg.pinv(X_b.T @ X_b + lam * I) @ X_b.T @ y
    return theta

def lasso_regression(X, y, lam=0.1, max_iter=500, tol=1e-4):
    """
    Coordinate descent Lasso implementation.
    Runs on subset for large datasets.
    """
    X_b = add_bias(X)
    n, d = X_b.shape
    theta = np.zeros(d)

    for iteration in range(max_iter):
        theta_old = theta.copy()
        for j in range(d):
            residual = y - (X_b @ theta) + theta[j] * X_b[:, j]
            rho = np.sum(X_b[:, j] * residual)
            z = np.sum(X_b[:, j] ** 2)
            if j == 0:
                theta[j] = rho / z
            else:
                if rho < -lam / 2:
                    theta[j] = (rho + lam / 2) / z
                elif rho > lam / 2:
                    theta[j] = (rho - lam / 2) / z
                else:
                    theta[j] = 0.0
        if np.max(np.abs(theta - theta_old)) < tol:
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

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    # Load your processed Retail dataset
    df = pd.read_csv("../data/processed_data.csv")  # adjust path if needed
    target_col = "avg_purchase_value"

    # Only numeric columns
    X = df.drop(columns=[target_col]).select_dtypes(include=[np.number]).values
    y = df[target_col].values

    # ---------- Train/Test Split ----------
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    split = int(0.8 * len(X))
    train_idx, test_idx = indices[:split], indices[split:]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # project root
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    RESULTS_DIR = os.path.join(BASE_DIR, "results")
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Linear Regression
    theta_lin = linear_regression(X_train, y_train)
    with open(os.path.join(MODELS_DIR, "regression_model1.pkl"), "wb") as f:
        pickle.dump(theta_lin, f)

    # 2. Polynomial Regression
    X_poly_train = polynomial_features(X_train, degree=2)
    theta_poly = linear_regression(X_poly_train, y_train)
    with open(os.path.join(MODELS_DIR, "regression_model2.pkl"), "wb") as f:
        pickle.dump((theta_poly, 2), f)

    # 3. Ridge Regression
    theta_ridge = ridge_regression(X_train, y_train, lam=10)
    with open(os.path.join(MODELS_DIR, "regression_model3.pkl"), "wb") as f:
        pickle.dump(theta_ridge, f)

    # 4. Lasso Regression (use subset if dataset is huge)
    subset_size = min(50000, X_train.shape[0])
    theta_lasso = lasso_regression(X_train[:subset_size], y_train[:subset_size], lam=0.1)
    with open(os.path.join(MODELS_DIR, "regression_model4.pkl"), "wb") as f:
        pickle.dump(theta_lasso, f)

    # ---------- Evaluate on Test Data ----------
    mse1, rmse1, r21 = evaluate(X_test, y_test, theta_lin)
    X_poly_test = polynomial_features(X_test, degree=2)
    mse2, rmse2, r22 = evaluate(X_poly_test, y_test, theta_poly)
    mse3, rmse3, r23 = evaluate(X_test, y_test, theta_ridge)
    mse4, rmse4, r24 = evaluate(X_test, y_test, theta_lasso)

    # Choose best based on R²
    best_candidates = [
        (r21, "Linear Regression", theta_lin, mse1, rmse1),
        (r22, "Polynomial Regression", theta_poly, mse2, rmse2),
        (r23, "Ridge Regression", theta_ridge, mse3, rmse3),
        (r24, "Lasso Regression", theta_lasso, mse4, rmse4)
    ]
    best = max(best_candidates, key=lambda x: x[0])
    best_r2, best_name, best_theta, best_mse, best_rmse = best

    # Save final model
    with open(os.path.join(MODELS_DIR, "regression_model_final.pkl"), "wb") as f:
        pickle.dump(best_theta, f)

    # Save metrics in exact format
    metrics_path = os.path.join(RESULTS_DIR, "train_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write("Regression Metrics:\n")
        f.write(f"Mean Squared Error (MSE): {best_mse:.2f}\n")
        f.write(f"Root Mean Squared Error (RMSE): {best_rmse:.2f}\n")
        f.write(f"R-squared (R²) Score: {best_r2:.2f}\n")

    print("Models trained and saved. Best:", best[1], "R² =", best[0])

    # Print metrics
    print("Regression Metrics:")
    print(f"Mean Squared Error (MSE): {best_mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {best_rmse:.2f}")
    print(f"R-squared (R²) Score: {best_r2:.2f}")
