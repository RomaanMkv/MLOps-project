import numpy
import great_expectations as gx
import pandas as pd
import pandas as pd
from omegaconf import DictConfig
import os
import yaml

from sklearn.preprocessing import OneHotEncoder
from datetime import datetime
from sklearn.preprocessing import StandardScaler

from zenml.client import Client
import zenml


def sample_data(cfg: DictConfig) -> None:
    
    # Read the data file
    data_url = cfg.data.path + cfg.data.raw_data_name + str(cfg.data.version) + cfg.data.data_format

    df = pd.read_csv(data_url)

    # Take a sample
    sample_size = cfg.data.sample_size
    sample_df = df.sample(frac=sample_size, random_state=42).reset_index(drop=True)

    # Create samples folder if it doesn't exist
    sample_folder = cfg.data.sample_folder
    os.makedirs(sample_folder, exist_ok=True)

    # Save the sample to the samples folder
    sample_path = os.path.join(sample_folder, "sample.csv")
    sample_df.to_csv(sample_path, index=False)


class DataValidationException(Exception):
    pass


def validate_initial_data(cfg: DictConfig) -> None:
    context = gx.get_context(project_root_dir = cfg.gx.project_root_dir)

    results = context.run_checkpoint(checkpoint_name="initial_data_validation_checkpoint_data")

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


def extract_data(base_path):
    version_file_path = base_path + '/configs/data_version.yaml'
    version = 0
    with open(version_file_path, 'r') as file:
        data = yaml.safe_load(file)
        version = data.get('version')
    
    df_file_path = base_path + '/data/samples/sample.csv'
    df = pd.read_csv(df_file_path)

    return df, version


def preprocess_data(data):
    def encode(df):
        categorical = ["flat_type", "storey_range", "flat_model"]

        encoder = OneHotEncoder(sparse_output=False)
        encoded_data = encoder.fit_transform(df[categorical])
        encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical))
        df = df.drop(columns=categorical).join(encoded_df)
        return df
    
    def convert_time_columns(df):
        reference_date = datetime(1966, 1, 1)
        df['month'] = pd.to_datetime(df['month'], format='%Y-%m')
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
    
    def scale_columns(df):
        to_be_scaled = ['floor_area_sqm', 'month', 'lease_commence_date', 'remaining_lease']
        scaler = StandardScaler()
        df[to_be_scaled] = scaler.fit_transform(df[to_be_scaled])
        return df
    
    def get_coordinates(df):
        def create_address_string(row):
            return f"{row['town']}, {row['street_name']}, block {row['block']}, Singapore"

        def make_full_address(df):
            df['full_address'] = df.apply(create_address_string, axis=1)
            df = df.drop(columns=['town', 'block', 'street_name'])
            return df
        
        coord_df = pd.read_csv("data/coordinates.csv", index_col='full_address')

        def get_coordinate(full_addr):
            try:
                result = coord_df.loc[full_addr]
                return numpy.float64(result['latitude']), numpy.float64(result['longitude'])
            except KeyError:
                return numpy.nan, numpy.nan
        
        df = make_full_address(df)
        df[['latitude', 'longitude']] = df['full_address'].apply(lambda addr: pd.Series(get_coordinate(addr)))
        df = df.drop(columns= 'full_address')

        return df
    

    data = encode(data)
    data = convert_time_columns(data)
    data = scale_columns(data)
    data = get_coordinates(data)
    data['latitude'] = pd.to_numeric(data['latitude'], errors='coerce')
    data['longitude'] = pd.to_numeric(data['longitude'], errors='coerce')
    data = data.dropna()

    X = data.drop(columns=['resale_price'])
    y = data['resale_price']
    

    return X, y

def validate_features(X, y):
    if not os.path.exists('data/preprocessed'):
        os.makedirs('data/preprocessed')

    X.to_csv('data/preprocessed/X.csv')
    y.to_csv('data/preprocessed/y.csv')

    context = gx.get_context(project_root_dir = 'services')

    results = context.run_checkpoint(checkpoint_name="preprocessed_data_validation_checkpoint_data")

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

# df, version = extract_data("/home/roman/MLOps/MLOps-project")
# X, y = preprocess_data(df)
# validate_features(X, y)



