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
CONFIG_PATH="configs"  # Ensure this path is correct and relative to your script's location
CONFIG_NAME="config"

# Function to take a data sample
take_data_sample() {
    echo "Taking a data sample..."
    python -c "
from hydra import initialize, compose
from omegaconf import OmegaConf
from src.data import sample_data

if __name__ == '__main__':
    # Initialize Hydra and compose the configuration
    with initialize(version_base=None, config_path='$CONFIG_PATH'):
        cfg = compose(config_name='$CONFIG_NAME')
        
        sample_data(cfg)
"
}

# Call the function
take_data_sample

# Function to validate the data sample
validate_data_sample() {
    echo "Validating the data sample..."
    python -c "
from hydra import initialize, compose
from src.data import validate_initial_data

if __name__ == '__main__':
    try:
        # Initialize Hydra and compose the configuration
        with initialize(version_base=None, config_path='$CONFIG_PATH'):
            cfg = compose(config_name='$CONFIG_NAME')
            
            validate_initial_data(cfg)
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
    git add data/samples/sample.csv.dvc data/samples/gitgnore
    git commit -m 'add data sample'
    git push
    dvc push
    echo "Data sample versioned successfully."
}

if validate_data_sample; then
    echo "Data validation passed."
    version_data_sample
else
    echo "Data validation failed. Data sample not versioned."
fi
