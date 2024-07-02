import os
from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.bash import BashOperator
from airflow.sensors.external_task import ExternalTaskSensor
from datetime import datetime, timedelta
import pendulum
from airflow.utils.state import DagRunState

# Define the base path using an environment variable or default to the current directory
project_base_path = os.getenv('PROJECT_BASE_PATH', os.getcwd())

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
    'data_prepare_dag',  # DAG ID
    default_args=default_args,
    description='A data preparation DAG that runs after data extraction DAG',  # Description of the DAG
    schedule='*/5 * * * *',  # Schedule to run every 5 minutes
    start_date=pendulum.today('UTC').add(days=-1),  # Start date for the DAG
    catchup=False,  # Whether to catch up on missed runs
)

# ExternalTaskSensor to wait for the completion of data_extract_dag
wait_for_data_extraction = ExternalTaskSensor(
    task_id='wait_for_data_extraction',
    external_dag_id='data_extract_dag',  # DAG ID of the data extraction pipeline
    external_task_id=None,  # Wait for the entire DAG to complete
    allowed_states=[DagRunState.SUCCESS],  # Only proceed if the data extraction pipeline succeeds
    failed_states=[DagRunState.FAILED],  # Do not proceed if the data extraction pipeline fails
    execution_delta=timedelta(minutes=5),  # Wait for the same schedule interval
    mode='reschedule',  # Reschedule the task if not yet complete
    dag=dag,
)

# Define the task to run the ZenML pipeline
run_zenml_pipeline = BashOperator(
    task_id='run_zenml_pipeline',
    bash_command=f"""
    cd {project_base_path}
    {project_base_path}/venv/bin/python pipelines/data_prepare.py -prepare_data_pipeline
    """,
    dag=dag,
)

# Set the task dependencies
wait_for_data_extraction >> run_zenml_pipeline

if __name__ == "__main__":
    dag.test()
