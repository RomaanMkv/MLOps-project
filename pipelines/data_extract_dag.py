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
    'data_extract_dag22',  # DAG ID
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
import hydra
from omegaconf import DictConfig
from src.data import sample_data

@hydra.main(config_path='$CONFIG_PATH', config_name='$CONFIG_NAME')
def run_sample_data(cfg: DictConfig) -> None:
    sample_data(cfg)

run_sample_data()
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
    echo "Data sample versioned successfully."
    # Store the DVC version in the configuration file
    echo 'version: '$(dvc status -c data/samples/sample.csv) > ./configs/data_version.yaml
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
