import os
from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

class DataValidationException(Exception):
    pass

# Define the base path using an environment variable or default to the current directory
project_base_path = os.getenv('PROJECT_BASE_PATH', os.getcwd())

config_path = "configs"
config_name = "config"

# Define default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
dag = DAG(
    'data_extract_dag',  # DAG ID
    default_args=default_args,
    description='A simple data extraction and validation DAG',  # Description of the DAG
    schedule_interval='*/5 * * * *',  # Schedule to run every 5 minutes
    start_date=days_ago(1),  # Start date for the DAG
    catchup=False,  # Whether to catch up on missed runs
)

# Define the task with multiple commands
sample_data = BashOperator(
    task_id='sample_data',
    bash_command=f"""
    cd {project_base_path}
    echo "Taking a data sample..."
    {project_base_path}/venv/bin/python -c "
from hydra import initialize, compose
from omegaconf import OmegaConf
from src.data import sample_data

if __name__ == '__main__':
    # Initialize Hydra and compose the configuration
    with initialize(version_base=None, config_path='$CONFIG_PATH'):
        cfg = compose(config_name='$CONFIG_NAME')
        
        sample_data(cfg)
    "
    """,
    env={'CONFIG_PATH': config_path, 'CONFIG_NAME': config_name},  # Setting environment variables
    dag=dag,
)

validate_data = BashOperator(
    task_id='validate_data',
    bash_command=f"""
    cd {project_base_path}
    {project_base_path}/venv/bin/python -c "
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
    """,
    env={'CONFIG_PATH': config_path, 'CONFIG_NAME': config_name},  # Setting environment variables
    dag=dag,
)

version_data = BashOperator(
    task_id='version_data',
    bash_command=f"""
    cd {project_base_path}
    echo "Versioning the data sample..."
    dvc add data/samples/sample.csv
    
    # Read the current version number from config.yaml
    current_version=$(grep 'sample_version' ./configs/config.yaml | awk -F ': ' '{{print $2}}' | tr -d '[:space:]')

    # Check if the current_version is correctly extracted
    if [[ -z "$current_version" ]]; then
    echo "Failed to extract the current version from config.yaml"
    exit 1
    fi

    git push -d origin "V$current_version"
    git tag -d "V$current_version"

    git add data/samples/sample.csv.dvc data/samples/gitgnore
    git commit -m 'add data sample'
    git push
    dvc push
    echo "Data sample versioned successfully."

    git tag -a "V$current_version" -m "A new version of the data 'V$current_version'"
    git push --tags

    # Update the version in the YAML file
    echo "version: $current_version" > configs/data_version.yaml

    """,
    dag=dag,
)


# Task to load the sample to the data store (dvc push)
load_sample = BashOperator(
    task_id='load_sample',
    bash_command=f"""
    cd {project_base_path}
    echo "Pushing data sample to remote storage..."
    dvc push
    echo "Data sample pushed successfully."
    """,
    dag=dag,
)

# Set the task dependencies
sample_data >> validate_data >> version_data >> load_sample

if __name__ == "__main__":
    dag.test()
