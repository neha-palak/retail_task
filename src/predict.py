import argparse
import os
import pickle
import numpy as np
import pandas as pd
from data_preprocessing import preprocess_data
from train_model import add_bias, polynomial_features


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def evaluate(y_true, y_pred):
    """Evaluate regression metrics directly on predictions"""
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot
    return mse, rmse, r2


def main(args):
    # Paths
    model_path = args.model_path or os.path.join(BASE_DIR, "models", "regression_model_final.pkl")
    data_path = args.data_path or os.path.join(BASE_DIR, "data", "processed_data.csv")
    metrics_output_path = args.metrics_output_path or os.path.join(BASE_DIR, "results", "train_metrics.txt")
    predictions_output_path = args.predictions_output_path or os.path.join(BASE_DIR, "results", "train_predictions.csv")

    # Load processed dataset
    df = pd.read_csv(data_path)
    target_col = "avg_purchase_value"
    y = df[target_col].values
    X = df.drop(columns=[target_col]).values

    # Load saved model
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    if isinstance(model, tuple):  # Polynomial regression
        theta, degree = model
        X = polynomial_features(X, degree)
    else:
        theta = model
    
    # Predictions
    degree = 2
    X = polynomial_features(X, degree)
    X_b = add_bias(X)
    y_pred = X_b @ theta

    # Save predictions
    os.makedirs(os.path.dirname(predictions_output_path), exist_ok=True)
    pd.DataFrame(y_pred, columns=["prediction"]).to_csv(predictions_output_path, index=False)

    # Evaluate
    mse, rmse, r2 = evaluate(y, y_pred)

    os.makedirs(os.path.dirname(metrics_output_path), exist_ok=True)
    with open(metrics_output_path, "w") as f:
        f.write("Regression Metrics:\n")
        f.write(f"Mean Squared Error (MSE): {mse:.4f}\n")
        f.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}\n")
        f.write(f"R-squared (R^2): {r2:.4f}\n")

    print(f" Predictions saved in {predictions_output_path}")
    print(f" Metrics saved in {metrics_output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate saved regression model")
    parser.add_argument("--model_path", type=str, help="Path to saved model pickle file")
    parser.add_argument("--data_path", type=str, help="Path to processed data CSV")
    parser.add_argument("--metrics_output_path", type=str, help="Where to save evaluation metrics")
    parser.add_argument("--predictions_output_path", type=str, help="Where to save predictions")

    args = parser.parse_args()
    main(args)
