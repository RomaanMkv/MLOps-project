import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import GridSearchCV
from zenml.client import Client
import pandas as pd
import mlflow
import mlflow.sklearn
import importlib
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from mlflow.tracking import MlflowClient
import matplotlib.pyplot as plt
import os
import pickle

def load_features(version, name = "features_target", fraction=0.01):
    client = Client()

    # Retrieve the list of artifacts for the given name and version
    artifacts = client.list_artifact_versions(name=name, tag=version, sort_by="version").items

    # Ensure that the artifacts list is not empty and reverse it to get the latest artifact first
    if not artifacts:
        raise ValueError(f"No artifacts found for name '{name}' with version '{version}'")
    
    artifacts.reverse()

    latest_artifact = artifacts[0].load()

    latest_artifact = latest_artifact.sample(frac=fraction, random_state=1)

    # Split the loaded artifact into X and y
    X = latest_artifact.iloc[:, :-1]
    y = latest_artifact.iloc[:, -1]

    return X, y


def log_metadata(cfg, gs, X_train, y_train, X_test, y_test):

    cv_results = pd.DataFrame(gs.cv_results_).filter(regex=r'std_|mean_|param_').sort_index(axis=1)
    best_metrics_values = [result[1][gs.best_index_] for result in gs.cv_results_.items()]
    best_metrics_keys = [metric for metric in gs.cv_results_]
    best_metrics_dict = {k:v for k,v in zip(best_metrics_keys, best_metrics_values) if 'mean' in k or 'std' in k}

    # print(100*'#')
    # print(cv_results, cv_results.columns)

    params = best_metrics_dict

    df_train = pd.concat([X_train, y_train], axis = 1)
    df_test = pd.concat([X_test, y_test], axis = 1)

    experiment_name = cfg.model.model_name + "_" + cfg.experiment_name 

    try:
        # Create a new MLflow Experiment
        experiment_id = mlflow.create_experiment(name=experiment_name)
    except mlflow.exceptions.MlflowException as e:
        experiment_id = mlflow.get_experiment_by_name(name=experiment_name).experiment_id # type: ignore
    
    print("experiment-id : ", experiment_id)

    cv_evaluation_metric = cfg.model.cv_evaluation_metric
    run_name = "_".join([cfg.run_name, cfg.model.model_name, cfg.model.evaluation_metric, str(params[cv_evaluation_metric]).replace(".", "_")]) # type: ignore
    print("run name: ", run_name)

    # Parent run
    with mlflow.start_run(run_name = run_name, experiment_id = experiment_id) as run:

        df_train_dataset = mlflow.data.pandas_dataset.from_pandas(df = df_train, targets = cfg.data.target_cols[0]) # type: ignore
        df_test_dataset = mlflow.data.pandas_dataset.from_pandas(df = df_test, targets = cfg.data.target_cols[0]) # type: ignore
        mlflow.log_input(df_train_dataset, "training")
        mlflow.log_input(df_test_dataset, "testing")

        # Log the hyperparameters
        mlflow.log_params(gs.best_params_)

        # Log the performance metrics
        mlflow.log_metrics(best_metrics_dict)

        # Set a tag that we can use to remind ourselves what this run was for
        mlflow.set_tag(cfg.model.tag_key, cfg.model.tag_value)

        # Infer the model signature
        signature = mlflow.models.infer_signature(X_train, gs.predict(X_train))

        # Log the model
        model_info = mlflow.sklearn.log_model(
            sk_model = gs.best_estimator_,
            artifact_path = cfg.model.artifact_path,
            signature = signature,
            input_example = X_train.iloc[0].to_numpy(),
            registered_model_name = cfg.model.model_name,
            pyfunc_predict_fn = cfg.model.pyfunc_predict_fn
        )

        # save the best model
        client = mlflow.client.MlflowClient()
        client.set_model_version_tag(name = cfg.model.model_name, version=model_info.registered_model_version, key="source", value="best_Grid_search_model")
        client.set_registered_model_alias(name=cfg.model.model_name, alias=cfg.model.best_model_alias, version=model_info.registered_model_version)


        for index, result in cv_results.iterrows():

            child_run_name = "_".join(['child', run_name, str(index)]) # type: ignore
            with mlflow.start_run(run_name=child_run_name, experiment_id=experiment_id, nested=True) as child_run: #, tags=best_metrics_dict):
                ps = result.filter(regex='param_').to_dict()
                ms = result.filter(regex='mean_').to_dict()
                stds = result.filter(regex='std_').to_dict()

                # Remove param_ from the beginning of the keys
                ps = {k.replace("param_",""):v for (k,v) in ps.items()}

                mlflow.log_params(ps)
                # Cast values that don't have digits after the decimal point to integers
                for key, value in ps.items():
                    if isinstance(value, float) and value.is_integer():
                        ps[key] = int(value)
                
                print(f'!ps={ps}')

                mlflow.log_metrics(ms)
                mlflow.log_metrics(stds)

                # We will create the estimator at runtime
                module_name = cfg.model.module_name # e.g. "sklearn.linear_model"
                class_name  = cfg.model.class_name # e.g. "LogisticRegression"

                # Load "module.submodule.MyClass"
                class_instance = getattr(importlib.import_module(module_name), class_name)
                
                estimator = class_instance(**ps)
                estimator.fit(X_train, y_train)
                
                signature = mlflow.models.infer_signature(X_train, estimator.predict(X_train))

                model_info = mlflow.sklearn.log_model(
                    sk_model = estimator,
                    artifact_path = cfg.model.artifact_path,
                    signature = signature,
                    input_example = X_train.iloc[0].to_numpy(),
                    registered_model_name = cfg.model.model_name
                )

                client = mlflow.client.MlflowClient()

                model_uri = model_info.model_uri
                loaded_model = mlflow.sklearn.load_model(model_uri=model_uri)

                y_pred = loaded_model.predict(X_test) # type: ignore
        
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)

                mlflow.log_metric("mean_squared_error", mse)
                mlflow.log_metric("root_mean_squared_error", rmse)
                mlflow.log_metric("r_2_score", r2)

                print(f'mse={mse}')
                print(f'rmse={rmse}')
                print(f'r2={r2}')

                # Plot and log performance charts
                fig, ax = plt.subplots()
                ax.scatter(y_test, y_pred, edgecolors=(0, 0, 0))
                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
                ax.set_xlabel('Measured')
                ax.set_ylabel('Predicted')
                ax.set_title(f'Performance of model: {child_run_name}')

                plot_name = "performance_plot.png"
                mlflow.log_figure(fig, plot_name)
    
                # Download artifacts
                dst_path = f"results/{child_run_name}"
                if not os.path.exists(dst_path):
                    os.makedirs(dst_path)

                mlflow.artifacts.download_artifacts(artifact_uri=f'{child_run.info.artifact_uri}/{plot_name}', dst_path=dst_path)

    version_metrics = get_model_versions_with_metrics(cfg.model.model_name, "root_mean_squared_error")
    sorted_versions = sort_versions_by_metric(version_metrics)
    assign_aliases_to_models(cfg.model.model_name, sorted_versions, cfg.model.ordinal_model_alias)


def train(X_train, y_train, cfg):

    # Define the model hyperparameters
    params = cfg.model.params

    # Train the model
    module_name = cfg.model.module_name # e.g. "sklearn.linear_model"
    class_name  = cfg.model.class_name # e.g. "LogisticRegression"

    # We will create the estimator at runtime
    import importlib

    # Load "module.submodule.MyClass"
    class_instance = getattr(importlib.import_module(module_name), class_name)

    estimator = class_instance()

    # Grid search with cross validation
    from sklearn.model_selection import KFold
    cv = KFold(n_splits=cfg.model.folds, random_state=cfg.random_state, shuffle=True)

    param_grid = dict(params)

    scoring = list(cfg.model.metrics.values()) # ['balanced_accuracy', 'f1_weighted', 'precision', 'recall', 'roc_auc']

    evaluation_metric = cfg.model.evaluation_metric

    gs = GridSearchCV(
        estimator = estimator,
        param_grid = param_grid,
        scoring = scoring,
        n_jobs = cfg.cv_n_jobs,
        refit = evaluation_metric,
        cv = cv,
        verbose = 3,
        return_train_score = True
    )

    gs.fit(X_train, y_train)

    return gs


def get_model_versions_with_metrics(model_name, metric_name="root_mean_squared_error"):
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")
    version_metrics = []

    for version in versions:
        run_id = version.run_id
        metrics = client.get_run(run_id).data.metrics
        if metric_name in metrics:
            version_metrics.append((version.version, metrics[metric_name]))
        else:
            print(f"Metric {metric_name} not found for version {version.version}")

    return version_metrics


def sort_versions_by_metric(version_metrics):
    return sorted(version_metrics, key=lambda x: x[1])


def assign_aliases_to_models(model_name, sorted_versions, ordinal_alias):
    client = MlflowClient()
    
    # Assign 'champion' alias to the model with the smallest metric
    # champion_version = sorted_versions[0][0]
    # client.set_registered_model_alias(model_name, "champion", champion_version)
    # print(f"Assigned alias 'champion' to version {champion_version}")

    # Assign 'challenger1', 'challenger2', etc. to the remaining models
    for i, (version, metric) in enumerate(sorted_versions):
        alias = f"{ordinal_alias}{i}"
        client.set_registered_model_alias(model_name, alias, version)
        print(f"Assigned alias '{alias}' to version {version}")


def retrieve_model_with_alias(model_name, model_alias = "champion") -> mlflow.pyfunc.PyFuncModel:

    best_model:mlflow.pyfunc.PyFuncModel = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}@{model_alias}")

    # best_model
    return best_model


def retrieve_model_with_version(model_name, model_version = "v1") -> mlflow.pyfunc.PyFuncModel:

    best_model:mlflow.pyfunc.PyFuncModel = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")

    # best_model
    return best_model


def save_best_model(cfg, model_alias = "champion"):
    model_uri = f"models:/{cfg.model.model_name}@{model_alias}"
    sklearn_model = mlflow.sklearn.load_model(model_uri=model_uri)

    save_dir = f'models/{cfg.model.model_name}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    pickle_save_path = os.path.join(save_dir, f"{cfg.model.model_name}_{model_alias}.pkl")
    
    # Save the model locally using pickle
    with open(pickle_save_path, 'wb') as f:
        pickle.dump(sklearn_model, f)
    
    print(f"Model saved locally as pickle at {pickle_save_path}")

