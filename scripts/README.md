# Scripts Folder 

## Overview
The `scripts` folder contains various shell scripts used for automating tasks, managing environments, and executing commands within the MLOps project. Each script performs specific functions related to project setup, deployment, model management, and more.

## Scripts Description

### `deploy_docker.sh`
- **Purpose**: Builds and deploys a Docker container for the machine learning service.
- **Key Operations**:
  - Navigates to the project directory.
  - Builds a Docker image and runs a container.
  - Logs in to Docker and pushes the image to a Docker repository.

### `install_requirements.sh`
- **Purpose**: Sets up the project environment by installing required dependencies.
- **Key Operations**:
  - Creates a virtual environment if it does not exist.
  - Updates `.bashrc` with environment variables.
  - Activates the virtual environment.
  - Installs Python packages from `requirements.txt`.

### `mlflow_commands.sh`
- **Purpose**: Executes various MLflow commands for running different stages of the ML pipeline.
- **Key Operations**:
  - Runs MLflow commands for data extraction, transformation, model creation, evaluation, validation, and prediction.
  - Starts the MLflow server.

### `predict_samples.sh`
- **Purpose**: Executes predictions using different sample versions.
- **Key Operations**:
  - Navigates to the project directory.
  - Runs the prediction script `predict.py` for different versions of sample data.

### `run_airflow.sh`
- **Purpose**: Starts the Apache Airflow scheduler and webserver in daemon mode.
- **Key Operations**:
  - Navigates to the project directory.
  - Starts Airflow services and logs output to specified files.

### `stop_airflow.sh`
- **Purpose**: Stops running Apache Airflow services.
- **Key Operations**:
  - Navigates to the project directory.
  - Kills Airflow processes.
  - Optionally backfills Airflow DAGs.

### `test_data.sh`
- **Purpose**: Tests data sampling, validation, and versioning.
- **Key Operations**:
  - Takes a data sample using a Python script.
  - Validates the sampled data.
  - Versions the data sample using DVC (Data Version Control) if validation passes.

## Usage Guidelines
1. **Execution**: Run scripts from the command line. Ensure you have the necessary permissions and environment set up.
2. **Dependencies**: Ensure that any external tools or services referenced in the scripts (e.g., Docker, MLflow, Airflow) are installed and configured properly.
3. **Customization**: Modify paths and parameters in the scripts as needed to match your local setup or project requirements.
4. **Error Handling**: Review script output for any errors or warnings. Address issues as they arise to ensure smooth execution.

## Important Notes
- **Environment Variables**: Some scripts assume specific environment variables and project directory structures. Ensure these are correctly configured.

By following these guidelines, you can effectively use the scripts in this folder to manage and automate various tasks within the MLOps project.