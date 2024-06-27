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

# Check if virtual environment folder exists, if not, create one
if [ ! -d "venv" ]; then
    python3.11 -m venv venv
fi

# Activate the virtual environment
source venv/bin/activate

# Install the requirements
pip install -r requirements.txt
