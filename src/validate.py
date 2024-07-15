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

model_name = cfg.model.best_model_name

# You can sweep over challenger aliases using Hydra
model_alias = cfg.model.best_model_alias

model: mlflow.pyfunc.PyFuncModel = retrieve_model_with_alias(model_name, model_alias=model_alias)
client = mlflow.MlflowClient()
mv = client.get_model_version_by_alias(name=model_name, alias=model_alias)
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
    #target=TARGET_COLUMN  # Target column name
)

pred_test_wrapped = giskard_model.predict(giskard_dataset).raw_prediction
print('11111111111', pred_test_wrapped)


# # Save the results in `html` file
# scan_results_path = f"reports/validation_results_{model_name}_{model_version}_{dataset_name}_{version}.html"
# scan_results.to_html(scan_results_path)

# suite_name = f"test_suite_{model_name}_{model_version}_{dataset_name}_{version}"
# test_suite = giskard.Suite(name = suite_name)

# test1 = giskard.testing.test_rmse(model=giskard_model, dataset=giskard_dataset, threshold=40000)

# test_suite.add_test(test1)

# test_results = test_suite.run()
# if (test_results.passed):
#     print("Passed model validation!")
# else:
#     print("Model has vulnerabilities!")
