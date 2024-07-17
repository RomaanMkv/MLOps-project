import numpy as np
import great_expectations as gx
import pandas as pd
import pandas as pd
from omegaconf import DictConfig
import os
import yaml

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from zenml.client import Client
import zenml


def sample_data(cfg: DictConfig) -> None:
    
    # Read the data file
    data_url = cfg.sample_data.path + cfg.sample_data.raw_data_name + str(cfg.sample_data.sample_version) + cfg.sample_data.data_format

    df = pd.read_csv(data_url)

    # Take a sample
    sample_size = cfg.sample_data.sample_size
    sample_df = df.sample(frac=sample_size, random_state=42).reset_index(drop=True)

    # Create samples folder if it doesn't exist
    sample_folder = cfg.sample_data.sample_folder
    os.makedirs(sample_folder, exist_ok=True)

    # Save the sample to the samples folder
    sample_path = os.path.join(sample_folder, "sample.csv")
    sample_df.to_csv(sample_path, index=False)


class DataValidationException(Exception):
    pass


def validate_initial_data(cfg: DictConfig) -> None:
    context = gx.get_context(project_root_dir = cfg.val_in_data.project_root_dir)

    results = context.run_checkpoint(checkpoint_name=cfg.val_in_data.checkpoint_name)

    # Print detailed validation results
    print("Validation success:", results.success)
    for result in results["run_results"].values():
        validation_result = result["validation_result"]
        for res in validation_result["results"]:
            expectation = res["expectation_config"]["expectation_type"]
            success = res["success"]
            print(f"Expectation {expectation}: {'SUCCESS' if success else 'FAILURE'}")
            if not success:
                print(f"Details: {res['result']}")
        if not results.success:
            raise DataValidationException("Data validation failed for one or more expectations.")

    print("All expectations passed successfully.")


def extract_data(cfg: DictConfig, base_path = None):
    if base_path is None:
        base_path = os.path.expandvars("$PROJECT_BASE_PATH")
        
    version_file_path = base_path + cfg.extr_data.version_file_path
    version = 0
    with open(version_file_path, 'r') as file:
        data = yaml.safe_load(file)
        version = data.get('version')
    
    df_file_path = base_path + cfg.extr_data.df_file_path
    df = pd.read_csv(df_file_path)

    return df, str(version)


def preprocess_data(data, cfg: DictConfig, only_X = False):

    def convert_time_columns(df):
        reference_date = datetime.strptime(cfg.prepr_data.reference_date, "%Y-%m-%d")
        try:
            df['month'] = pd.to_datetime(df['month'], format='mixed')
        except ValueError:
            df['month'] = pd.to_datetime('2000-01-01')
        df['month_seconds'] = (df['month'] - reference_date).dt.total_seconds()
        df['lease_commence_date'] = pd.to_datetime(df['lease_commence_date'], format='%Y')
        df['lease_commence_date_seconds'] = (df['lease_commence_date'] - reference_date).dt.total_seconds()
        
        def calculate_lease_end(row):
            try:
                years, months = 0, 0
                parts = row['remaining_lease'].split()
                if 'years' in parts:
                    years = int(parts[parts.index('years') - 1])
                if 'months' in parts:
                    months = int(parts[parts.index('months') - 1])
                
                start_date = row['month']
                end_date = start_date + pd.DateOffset(years=years, months=months)
                return (end_date - reference_date).total_seconds()
            except Exception as e:
                print(f"Error processing row: {row}, error: {e}")
                return None
        
        df['remaining_lease_seconds'] = df.apply(calculate_lease_end, axis=1)
        df.drop(columns=['month', 'lease_commence_date', 'remaining_lease'], inplace=True)
        df.rename(columns={
            'month_seconds': 'month',
            'lease_commence_date_seconds': 'lease_commence_date',
            'remaining_lease_seconds': 'remaining_lease'
        }, inplace=True)
        
        return df

    def get_coordinates(df):
        def create_address_string(row):
            return f"{row['town']}, {row['street_name']}, block {row['block']}, Singapore"

        def make_full_address(df):
            df['full_address'] = df.apply(create_address_string, axis=1)
            df = df.drop(columns=['town', 'block', 'street_name'])
            return df
        
        coord_df = pd.read_csv(cfg.prepr_data.coordinates_file_path, index_col='full_address')

        def get_coordinate(full_addr):
            try:
                result = coord_df.loc[full_addr]
                return np.float64(result['latitude']), np.float64(result['longitude'])
            except KeyError:
                return np.nan, np.nan
        
        df = make_full_address(df)
        df[['latitude', 'longitude']] = df['full_address'].apply(lambda addr: pd.Series(get_coordinate(addr)))
        df = df.drop(columns='full_address')

        return df

    # Process the columns that need to be converted first
    data = convert_time_columns(data)
    data = get_coordinates(data)
    data['latitude'] = pd.to_numeric(data['latitude'], errors='coerce')
    data['longitude'] = pd.to_numeric(data['longitude'], errors='coerce')
    
    # Define the transformations
    categorical_features = list(cfg.prepr_data.categorical_features)
    numeric_features = list(cfg.prepr_data.numeric_features)
    coordinate_features = list(cfg.prepr_data.coordinate_features)
    
    # Pipeline for categorical features
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(sparse_output=False))
    ])
    
    # Pipeline for numeric features
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    # Combine all transformations
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features),
            ('num', numeric_transformer, numeric_features),
        ],
        remainder='passthrough'
    )
    
            
    if not only_X:
        X = data.drop(columns=[cfg.prepr_data.target_feature])
        y = data[[cfg.prepr_data.target_feature]]
    else:
        X = data
        y = None
    
    X_transformed = preprocessor.fit_transform(X)
    transformed_columns = (
        preprocessor.transformers_[0][1].named_steps['onehot'].get_feature_names_out(categorical_features).tolist() +
        numeric_features +
        coordinate_features
    )
    X = pd.DataFrame(X_transformed, columns=transformed_columns, index=X.index)

    columns_needed = list(cfg.prepr_data.columns_needed)
    # Retain only the columns that are in 'columns_needed'
    X = X[[col for col in X.columns if col in columns_needed]]

    # Add any missing columns from 'columns_needed' with all values set to 0
    for col in columns_needed:
        if col not in X.columns:
            X[col] = 0
    
    # Apply transformations
    X = X.fillna(X.mean())
    
    return X, y

def validate_features(X, y, cfg : DictConfig):
    if not os.path.exists(cfg.val_feat.data_path):
        os.makedirs(cfg.val_feat.data_path)

    X.to_csv(cfg.val_feat.X_path)
    y.to_csv(cfg.val_feat.y_path)

    context = gx.get_context(project_root_dir = cfg.val_feat.project_root_dir)

    results = context.run_checkpoint(checkpoint_name=cfg.val_feat.checkpoint_name)

    # Print detailed validation results
    print("Validation success:", results.success)
    for result in results["run_results"].values():
        validation_result = result["validation_result"]
        for res in validation_result["results"]:
            expectation = res["expectation_config"]["expectation_type"]
            success = res["success"]
            print(f"Expectation {expectation}: {'SUCCESS' if success else 'FAILURE'}")
            if not success:
                print(f"Details: {res['result']}")
        if not results.success:
            raise DataValidationException("Data validation failed for one or more expectations.")

    print("All expectations passed successfully.")

    return X, y


def load_features(X: pd.DataFrame, y: pd.DataFrame, version: str):
    features_target = pd.concat([X, y], axis=1)

    zenml.save_artifact(data=features_target, name="features_target", tags=[version])

    client = Client()

    artifacts = client.list_artifact_versions(name="features_target", tag=version, sort_by="version").items

    artifacts.reverse()

    latest_artifact = artifacts[0].load()
    
    return latest_artifact


def load_artifact(name: str, version: str) -> pd.DataFrame:
    client = Client()
    artifacts = client.list_artifact_versions(name=name, tag=version, sort_by="version").items
    artifacts.reverse()
    return artifacts[0].load()


def transform_data(df: pd.DataFrame, cfg: DictConfig, fraction=1) -> pd.DataFrame:
    raw_df, _ = extract_data(cfg=cfg)
    raw_df = raw_df.sample(frac=fraction, random_state=cfg.random_state)

    X = raw_df.drop(columns=[cfg.prepr_data.target_feature])
    combined_df = pd.concat([X, df], ignore_index=True)
    
    combined_df, _ = preprocess_data(data=combined_df,
                                     cfg=cfg,
                                     only_X=True
                                     )
    
    df = combined_df.iloc[len(X):]
    
    return df
    

    


# df, version = extract_data("/home/roman/MLOps/MLOps-project")
# X, y = preprocess_data(df)
# print(type(y))
# artifact = load_artifact('features_target', '1')
# print(artifact)




