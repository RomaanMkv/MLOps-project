import pandas as pd
from typing_extensions import Tuple, Annotated
from zenml import step, pipeline, ArtifactConfig
from data import preprocess_data, extract_data, load_features, validate_features
import os
from hydra import initialize, compose
from omegaconf import OmegaConf

BASE_PATH = os.path.expandvars("$PROJECT_BASE_PATH")
CONFIG_PATH = '../configs'
CONFIG_NAME = 'config'

@step(enable_cache=False)
def extract()-> Tuple[
                Annotated[pd.DataFrame,
                        ArtifactConfig(name="extracted_data", 
                                       tags=["data_preparation"]
                                       )
                        ],
                Annotated[str,
                        ArtifactConfig(name="data_version",
                                       tags=["data_preparation"])]
                    ]:
    
    with initialize(version_base=None, config_path=CONFIG_PATH):
        cfg = compose(config_name=CONFIG_NAME)
        df, version = extract_data(base_path=BASE_PATH, cfg=cfg)

    return df, version

@step(enable_cache=False)
def transform(df: pd.DataFrame)-> Tuple[
                    Annotated[pd.DataFrame, 
                            ArtifactConfig(name="input_features",
                                           tags=["data_preparation"])],
                    Annotated[pd.DataFrame,
                            ArtifactConfig(name="input_target", 
                                            tags=["data_preparation"])]
                                    ]:

    # Your data transformation code
    X, y = preprocess_data(df)

    return X, y

@step(enable_cache=False)
def validate(X:pd.DataFrame, 
             y:pd.DataFrame)->Tuple[
                    Annotated[pd.DataFrame, 
                            ArtifactConfig(name="valid_input_features",
                                           tags=["data_preparation"])],
                    Annotated[pd.DataFrame,
                            ArtifactConfig(name="valid_target",
                                           tags=["data_preparation"])]
                                    ]:

    X, y = validate_features(X, y)
    
    return X, y


@step(enable_cache=False)
def load(X:pd.DataFrame, y:pd.DataFrame, version: str)-> Tuple[
                    Annotated[pd.DataFrame, 
                            ArtifactConfig(name="features",
                                           tags=["data_preparation"])],
                    Annotated[pd.DataFrame,
                            ArtifactConfig(name="target",
                                           tags=["data_preparation"])]
                                    ]:
    
    load_features(X, y, version)

    return X, y


@pipeline()
def prepare_data_pipeline():
    df, version = extract()
    X, y = transform(df)
    X, y = validate(X, y)
    X, y = load(X, y, version)


if __name__=="__main__":
    run = prepare_data_pipeline()