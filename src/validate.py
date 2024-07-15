import giskard.testing
from data import extract_data, preprocess_data  # custom module
from model import retrieve_model_with_alias  # custom module
from utils import init_hydra  # custom module
import giskard
import mlflow
import os
import pandas as pd

BASE_PATH = os.path.expandvars("$PROJECT_BASE_PATH")
cfg = init_hydra()
version = cfg.test_data_version
df, version = extract_data(base_path=BASE_PATH, cfg=cfg)

df = df[:100]
print(df)

# Specify categorical columns and target column
TARGET_COLUMN = cfg.data.target_cols[0]
CATEGORICAL_COLUMNS = list(cfg.data.cat_cols)
dataset_name = cfg.data.dataset_name

# Wrap your Pandas DataFrame with giskard.Dataset (validation or test set)
giskard_dataset = giskard.Dataset(
    df=df,  # A pandas.DataFrame containing raw data (before pre-processing) and including ground truth variable.
    target=TARGET_COLUMN,  # Ground truth variable
    name=dataset_name,  # Optional: Give a name to your dataset
    cat_columns=CATEGORICAL_COLUMNS  # List of categorical columns. Optional, but improves quality of results if available.
)

model_names = ['random_forest', 'gradient_boosting']
model_alias = 'challenger'
rmse_scores = {}

for model_name in model_names:
    for i in range(1, 27):
        model: mlflow.pyfunc.PyFuncModel = retrieve_model_with_alias(model_name, model_alias=f'{model_alias}{i}')
        client = mlflow.MlflowClient()
        mv = client.get_model_version_by_alias(name=model_name, alias=f'{model_alias}{i}')
        model_version = mv.version

        def predict(raw_df):
            X, _ = preprocess_data(
                data=raw_df,
                cfg=cfg,
                only_X=True
            )
            X = X.astype('float64')
            return model.predict(X)
        
        # Create the Giskard model for the regression problem
        giskard_model = giskard.Model(
            model=predict,  # The prediction function
            model_type="regression",  # Model type: "classification" or "regression"
            name=model_name,  # Name of your model
            feature_names=df.columns,  # List of feature names used by the model
        )

        scan_results = giskard.scan(giskard_model, giskard_dataset)

        # Save the results in `html` file
        scan_results_path = f"reports/validation_results_{model_name}_{model_version}_{dataset_name}_{version}.html"
        scan_results.to_html(scan_results_path)

        suite_name = f"test_suite_{model_name}_{model_version}_{dataset_name}_{version}"
        test_suite = giskard.Suite(name = suite_name)
        test1 = giskard.testing.test_rmse(model=giskard_model, dataset=giskard_dataset, threshold=40000)
        test_suite.add_test(test1)
        test_results = test_suite.run()

        # def props(cls):   
        #     return [i for i in cls.__dict__.keys() if i[:1] != '_']
        
        # properties = props(test_results.results[0].result)
        # print('properties =====', properties)
        
        if test_results.passed:
            print(f"Model {model_name} version {model_version} passed validation!")
            
            rmse_score = test_results.results[0].result.metric  # Assuming the RMSE value is stored in `actual_value`
            
            # Store RMSE score with model details
            rmse_scores[(model_name, model_version)] = rmse_score

        else:
            print(f"Model {model_name} version {model_version} has vulnerabilities!")


if rmse_scores:
    # Find the model with the smallest RMSE
    best_model = min(rmse_scores, key=rmse_scores.get)
    best_model_name, best_model_version = best_model

    # Set alias 'validation_champion' to the model with the smallest RMSE
    client.set_registered_model_alias(best_model_name, version=best_model_version, alias='validation_champion')
    print(f"Set 'validation_champion' alias to model {best_model_name} version {best_model_version} with RMSE {rmse_scores[best_model]}")
else:
    print("No model passed the test suite")

