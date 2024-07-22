# pipelines Folder 

## Overview
The `pipelines` folder contains the pipeline definitions for automating workflows and tasks within the project. These pipelines are primarily defined using Apache Airflow and ZenML, and they facilitate the data extraction, preparation, and validation processes. This directory ensures that workflows are automated, repeatable, and version-controlled.

### dags
This is a symbolic link to the DAGs (Directed Acyclic Graphs) directory used by Apache Airflow. The `dags` folder contains the definitions of the workflows managed by Airflow.

### data_extract_dag.py
This script defines an Airflow DAG for data extraction. It includes tasks for sampling data, validating the data, versioning the data, and loading the sample to the data store.

### data_prepare_dag.py
This script defines an Airflow DAG for data preparation. It includes tasks for waiting for data extraction to complete and then running a ZenML pipeline for data preparation. 

### data_prepare.py
This script defines the steps and pipeline for data preparation using ZenML. It includes steps for extracting, transforming, validating, and loading data. 
