# `src` Folder

## Overview

The `src` folder contains the core scripts and modules for the MLOps project. These scripts are responsible for data processing, model training, evaluation, prediction, and validation. Below is a brief description of each file:

## File Descriptions

### `convert_to_json.py`
- **Purpose**: Converts a sample from a CSV file to a JSON format.
- **Details**: Reads a CSV file `data/preprocessed/X.csv`, takes the first row, converts it to a dictionary, and then to a JSON string.

### `data.py`
- **Purpose**: Handles data sampling, validation, extraction, preprocessing, and feature validation.
- **Details**: Contains several functions:
  - `sample_data(cfg)`: Samples data based on the configuration.
  - `validate_initial_data(cfg)`: Validates the initial data using Great Expectations.
  - `extract_data(cfg, base_path)`: Extracts data and version from a file.
  - `preprocess_data(data, cfg, only_X, scaler_version)`: Preprocesses the data, including converting time columns, getting coordinates, and applying transformations.
  - `validate_features(X, y, cfg)`: Validates features and target data.
  - `load_features(X, y, version)`: Loads features and targets, saving them as an artifact.
  - `load_artifact(name, version)`: Loads an artifact by name and version.
  - `transform_data(df, cfg, fraction)`: Transforms new data using existing preprocessing steps.

### `evaluate.py`
- **Purpose**: Evaluates the trained machine learning model.
- **Details**: Loads a model and data, makes predictions, and evaluates the model using MLflow's evaluation metrics.

### `main.py`
- **Purpose**: Entry point for training and evaluating the machine learning model.
- **Details**: 
  - Uses Hydra for configuration management.
  - Loads training and testing data, trains the model, logs metadata, and saves the best model.

### `model.py`
- **Purpose**: Handles model training, saving, and metadata logging.
- **Details**: Functions for training the model, logging metadata, and saving the best model.

### `predict.py`
- **Purpose**: Performs predictions using a trained model hosted on a local server.
- **Details**: 
  - Uses Hydra for configuration management.
  - Sends a POST request with the input data to the prediction service and prints the prediction results.

### `utils.py`
- **Purpose**: Utility functions for the project.
- **Details**: 
  - `init_hydra()`: Initializes Hydra configuration management.

### `validate_champion.py`
- **Purpose**: Validates the champion model using Giskard.
- **Details**: 
  - Samples data, extracts and preprocesses it, loads the model, and runs validation tests using Giskard.

### `validate.py`
- **Purpose**: Validates multiple models and sets the best model as the champion.
- **Details**: 
  - Extracts and preprocesses data, loads models, runs validation tests, and sets the best model as the champion.

## Usage

1. **Data Preparation**: Use `data.py` to sample, extract, and preprocess the data.
2. **Model Training**: Run `main.py` to train the model.
3. **Model Evaluation**: Use `evaluate.py` to evaluate the trained model.
4. **Predictions**: Use `predict.py` to perform predictions using the trained model.
5. **Validation**: Use `validate.py` and `validate_champion.py` to validate models and set the best model as the champion.


- Configuration management is handled using Hydra. Configuration files are located in the `../configs` directory.
- Ensure that all required Python packages are installed. 