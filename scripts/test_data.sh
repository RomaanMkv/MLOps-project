#!/bin/bash

# Function to navigate up to the MLOps-project directory
navigate_to_mlops_project() {
  current_dir=$(pwd)
  target_dir="MLOps-project"

  while [ "$(basename "$current_dir")" != "$target_dir" ]; do
    current_dir=$(dirname "$current_dir")
    if [ "$current_dir" == "/" ]; then
      echo "MLOps-project directory not found."
      return
    fi
  done

  cd "$current_dir" || return
  echo "Changed directory to: $current_dir"
}

# Call the function
navigate_to_mlops_project

# Define the paths
CONFIG_PATH="configs"
CONFIG_NAME="config"

# Function to take a data sample
take_data_sample() {
    echo "Taking a data sample..."
    python -c "
import hydra
from src.data import sample_data
from omegaconf import DictConfig

@hydra.main(config_path='$CONFIG_PATH', config_name='$CONFIG_NAME')
def run_sample_data(cfg: DictConfig) -> None:
    sample_data(cfg)

run_sample_data()
"
}

# Function to validate the data sample
validate_data_sample() {
    echo "Validating the data sample..."
    python -c "
import hydra
from src.data import validate_initial_data
from omegaconf import DictConfig

@hydra.main(config_path='$CONFIG_PATH', config_name='$CONFIG_NAME')
def run_validate_initial_data(cfg: DictConfig) -> None:
    validate_initial_data(cfg)

try:
    run_validate_initial_data()
except Exception as e:
    import sys
    sys.exit(1)
"
}

# Function to version the data sample
version_data_sample() {
    echo "Versioning the data sample..."
    # Add the sample file to DVC for versioning
    dvc add data/samples/sample.csv
    echo "Data sample versioned successfully."
}

# Main script execution
take_data_sample

if validate_data_sample; then
    echo "Data validation passed."
    version_data_sample
else
    echo "Data validation failed. Data sample not versioned."
fi
