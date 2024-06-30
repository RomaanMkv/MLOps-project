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

pkill -f "airflow scheduler"
pkill -f "airflow webserver"
airflow dags backfill \
            --start-date START_DATE \
            --end-date END_DATE \
            dag_id