# notebooks Folder

## Overview
The `notebooks` folder contains Jupyter notebooks used for various stages of the data science workflow, including exploratory data analysis, data preprocessing, model development, and validation. This directory is essential for prototyping, experimentation, and understanding the data and models. Notebooks are organized based on their specific purpose and stage in the workflow.

### business_data_understanding.ipynb
This notebook is dedicated to understanding the business context and the data. It includes:

- Data description and exploration
- Identification of data features that need cleaning
- Cleaning methods applied to get clean data
- Description of the data quantity and data types
- Analysis of unique values per feature
- Categorization of feature types (categorical, numerical, text)
- Distribution analysis of each data feature and the target
- Bivariate analysis to understand linear relationships between features and the target
- Preliminary feature selection based on relationships and contribution to model performance
- Data transformation methods for ML-ready datasets
- Preliminary data transformation and ML modeling

### data_preprocessing.ipynb
This notebook focuses on data preprocessing steps necessary to prepare the data for modeling. It includes:

- Data cleaning techniques
- Handling missing values
- Data transformation methods (e.g., one-hot encoding for categorical features)
- Preliminary data transformations to get an ML-ready dataset
- Data quality verification and validation

### model_playground.ipynb
This notebook is used for experimenting with different machine learning models. It includes:

- Initial model selection and training
- Evaluation of model performance using various metrics
- Comparison of different models to select the best-performing one
- Insights and conclusions from model experiments

### PoC.ipynb
This notebook serves as a proof-of-concept (PoC) for building an ML model. It includes:

- Preliminary data analysis and feature selection
- Building a simple ML model to test feasibility
- Evaluation of the PoC model performance
- Documentation of findings and next steps for further model development

### validation_expectations.ipynb
This notebook is dedicated to data validation and testing. It includes:

- Writing expectations using Great Expectations (GX) about the data coming from the source
- Validation of these expectations
- Ensuring data quality and consistency before further processing

