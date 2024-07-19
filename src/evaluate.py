import argparse
import mlflow.sklearn
from model import load_features
import pandas as pd
import giskard # do not remove

def evaluate_model(data_sample_version, model_name, model_alias):
    # Load the model using the provided alias
    model = mlflow.sklearn.load_model(model_uri=f"models:/{model_name}@{model_alias}")

    X_test, y_test = load_features(version=data_sample_version, fraction=0.2)

    # Perform evaluation
    y_pred = model.predict(X_test)
    # Add your evaluation metrics calculation here
    # For example, calculating RMSE, R2, etc.
    eval_data = pd.DataFrame(y_test)
    eval_data.columns = ["actual"]
    eval_data["predictions"] = y_pred

    results = mlflow.evaluate(
        data=eval_data,
        model_type="regressor",
        targets="actual",
        predictions="predictions",
        evaluators="default"
    )

    print(f"'{model_name}' model with '{model_alias}' alias has the following metrics:\n{results.metrics}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model")
    parser.add_argument('--data_sample_version', type=str, required=True, help='Version of the data sample to use for evaluation')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model to evaluate')
    parser.add_argument('--model_alias', type=str, required=True, help='Alias of the model to evaluate')
    
    args = parser.parse_args()
    
    evaluate_model(args.data_sample_version, args.model_name, args.model_alias)