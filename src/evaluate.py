import argparse
import mlflow.sklearn
from model import load_features
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def evaluate_model(data_sample_version, model_name, model_alias):
    # Load the model using the provided alias
    model = mlflow.sklearn.load_model(model_uri=f"models:/{model_name}@{model_alias}")

    X_test, y_test = load_features(version=data_sample_version, fraction=0.2)



    # Perform evaluation
    y_pred = model.predict(X_test)
    # Add your evaluation metrics calculation here
    # For example, calculating RMSE, R2, etc.
    print(f"Evaluating model version: {data_sample_version}, alias: {model_alias}")
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    mlflow.log_metric("mean_squared_error", mse)
    mlflow.log_metric("root_mean_squared_error", rmse)
    mlflow.log_metric("r_2_score", r2)

    print(f'mse={mse}')
    print(f'rmse={rmse}')
    print(f'r2={r2}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model")
    parser.add_argument('--data_sample_version', type=str, required=True, help='Version of the data sample to use for evaluation')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model to evaluate')
    parser.add_argument('--model_alias', type=str, required=True, help='Alias of the model to evaluate')
    
    args = parser.parse_args()
    
    evaluate_model(args.data_sample_version, args.model_name, args.model_alias)