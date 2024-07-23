![Test code workflow](https://github.com/RomaanMkv/MLOps-project/actions/workflows/test-code.yaml/badge.svg)
![Test model workflow](https://github.com/RomaanMkv/MLOps-project/actions/workflows/validate-model.yaml/badge.svg)

## Table of Contents

1. [Project Overview](#project-overview)
2. [Scope of the Project](#scope-of-the-project)
   - [Background](#background)
   - [Business Problem](#business-problem)
   - [Business Objectives](#business-objectives)
   - [ML Objectives](#ml-objectives)
3. [Success Criteria](#success-criteria)
   - [Business Success Criteria](#business-success-criteria)
   - [ML Success Criteria](#ml-success-criteria)
   - [Economic Success Criteria](#economic-success-criteria)
4. [Data Collection](#data-collection)
   - [Data Sources](#data-sources)
   - [Data Collection Report](#data-collection-report)
5. [Getting Started](#getting-started)
   - [Prerequisites](#prerequisites)
   - [Installation](#installation)
6. [Usage](#usage)
7. [Deploying Your Best Model](#deploying-your-best-model)
8. [Folder Descriptions](#folder-descriptions)
   - [api](#api)
   - [configs](#configs)
   - [data](#data)
   - [data_base](#data_base)
   - [docs](#docs)
   - [models](#models)
   - [notebooks](#notebooks)
   - [outputs](#outputs)
   - [pipelines](#pipelines)
   - [results](#results)
   - [scripts](#scripts)
   - [services](#services)
   - [sql](#sql)
   - [src](#src)

---

## Project Overview

This project focuses on predicting the resale prices of HDB (Housing & Development Board) flats in Singapore using historical transaction data from January 2017 to June 2024. The goal is to develop a regression model that can accurately forecast resale prices, enabling stakeholders such as buyers, sellers, and investors to make informed decisions. The project also aims to provide insights into the factors influencing resale prices and uncover trends in the housing market.

## Scope of the Project

### Background

The Housing & Development Board (HDB) is a key provider of affordable housing in Singapore. HDB flats are sold under a lease system, and a significant secondary market exists for these flats. Accurate prediction of resale prices is crucial for various stakeholders, including buyers, sellers, and investors, to make informed decisions in this market. The dataset includes detailed information on HDB flat resale transactions, covering various attributes such as town, flat type, storey range, floor area, and resale price.

### Business Problem

The business problem is the need for accurate and reliable predictions of HDB flat resale prices. Current methods often rely on market knowledge and simple comparative analysis, which may not adequately capture the complex factors influencing resale prices. This project seeks to leverage data-driven approaches to enhance the accuracy of price predictions and improve market segmentation. Accurate price forecasts will support better decision-making for buyers, sellers, and investors and facilitate more efficient market operations.

### Business Objectives

1. **Develop a Predictive Model:** Create a regression model that can accurately estimate the resale prices of HDB flats based on historical transaction data.
2. **Understand Price Influences:** Identify the key factors that influence HDB resale prices, such as flat type, location, floor area, and lease details.
3. **Identify Market Trends:** Uncover temporal and spatial trends in the HDB resale market to inform future business strategies and policy decisions.
4. **Improve Market Segmentation:** Use the model’s outputs to enhance market segmentation and support targeted marketing strategies.
5. **Geospatial Insights:** Integrate geospatial data to analyze the geographic distribution of resale prices and provide additional insights.

### ML Objectives

1. **Predict Resale Prices:** Build a regression model to predict the resale prices of HDB flats using features such as transaction month, town, flat type, block, street name, storey range, floor area, flat model, lease commencement date, and remaining lease.
2. **Feature Importance Analysis:** Determine which features have the most significant impact on resale prices.
3. **Analyze Temporal Trends:** Explore how resale prices change over time and identify any seasonal patterns.
4. **Geospatial Analysis:** Incorporate geographic data to analyze spatial trends and visualize the distribution of resale prices across different regions.
5. **Price Segmentation:** Develop methods to categorize properties based on predicted prices to support targeted marketing.

## Success Criteria

### Business Success Criteria

1. **Accurate Price Predictions:** Achieve an average prediction error of less than 5% compared to actual resale prices.
2. **Enhanced Decision-Making:** Improve stakeholders' decision-making capabilities regarding buying, selling, and investing in HDB flats.
3. **Effective Market Segmentation:** Utilize model outputs to identify and target distinct market segments effectively.
4. **Insightful Trend Analysis:** Provide valuable insights into market trends and patterns that can inform strategic decisions.
5. **Geospatial Capabilities:** Successfully incorporate geospatial data to visualize and analyze geographic trends in resale prices.

### ML Success Criteria

1. **Model Accuracy:** Achieve a Root Mean Squared Error (RMSE) of less than SGD 30,000 on the test dataset.
2. **Generalization:** Ensure the model performs consistently well on new, unseen data.
3. **Feature Insights:** Provide clear insights into the importance of different features in predicting resale prices.
4. **Temporal Stability:** Maintain model robustness across different time periods and market conditions.
5. **Geospatial Accuracy:** Accurately reflect spatial trends and correlations in the resale prices using geographic data.
6. **Model Interpretability:** Ensure that stakeholders can understand and trust the model’s predictions.

### Economic Success Criteria

1. **Cost Efficiency:** Reduce reliance on manual market analysis and consultancy, leading to significant cost savings.
2. **Increased Revenue:** Leverage accurate price predictions and market segmentation to enhance sales and marketing effectiveness, driving revenue growth.
3. **Positive ROI:** Achieve a positive return on investment by ensuring the financial benefits from the project exceed the costs.
4. **Market Competitiveness:** Strengthen HDB's position in the housing market by providing valuable pricing insights and improving service offerings.
5. **Resource Allocation:** Optimize the allocation of resources in marketing and sales based on model insights, leading to better utilization of financial and human resources.

## Data Collection

### Data Sources

- **Primary Data Source:** The dataset is obtained from [data.gov.sg](https://data.gov.sg/), Singapore’s open data portal. It includes records of HDB flat resale transactions from January 2017 to June 2024.
- **Geospatial Data:** Additional geospatial data can be sourced using services like Google Maps API and Singapore’s OneMap API to convert street names into latitude and longitude coordinates.

### Data Collection Report

| Aspect | Description |
|--------|-------------|
| **Data Source** | The primary data source is data.gov.sg, which provides comprehensive records of HDB resale transactions. |
| **Data Type** | The dataset includes numerical data (e.g., floor area in sqm, resale price) and categorical data (e.g., town, flat type, block, street name, storey range, flat model, lease commencement date, remaining lease). |
| **Data Size** | The dataset contains approximately 181,000 rows and 11 columns, covering transactions over a period from January 1, 2017, to June 1, 2024. |
| **Data Collection Method** | The data was downloaded in CSV format from data.gov.sg. For spatial analysis, geocoding services will be used to convert addresses into geographic coordinates. |

## Getting Started

### Prerequisites

- Python 3.x
- Jupyter Notebook or any other Python IDE
- Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn, and others as specified in the requirements file.

### Installation

1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/RomaanMkv/MLOps-project.git
   ```
2. Install the required Python libraries:
   ```bash
   bash scripts/install_requirements.sh
   ```

## Usage

1. Load and explore the dataset to understand its structure and contents.
2. Preprocess the data to handle missing values, convert categorical variables to numerical format, and perform feature scaling.
3. Train the regression model using the training data.
4. Evaluate the model's performance on the test data using metrics like RMSE.
5. Analyze feature importance and interpret the model's predictions.
6. Incorporate geospatial data for spatial analysis and visualization of resale price trends.
7. Use the model to predict resale prices and derive actionable insights for stakeholders.

## Deploying Your Best Model

#### 1. Deploying Your Best Model using Flask API

To deploy your trained model using Flask, follow these steps:

1. **Ensure Dependencies**:
   Ensure that your environment matches the model's required dependencies. If there are mismatches, you can fix them using:
   ```bash
   mlflow.pyfunc.get_model_dependencies(model_uri)
   ```
   This command fetches the model's environment and installs the necessary dependencies.

2. **Run the Flask API**:
   Start the Flask server by running the following command:
   ```bash
   python3 api/app.py
   ```
   The server will start and you can access the API at `http://127.0.0.1:5001`.

3. **Test the API**:
   You can test the API endpoint with a sample request:
   ```bash
   curl -X POST -H "Content-Type: application/json" -d '{"flat_type_1 ROOM": 0.0, "flat_type_2 ROOM": 0.0, "flat_type_3 ROOM": 1.0, "flat_type_4 ROOM": 0.0, "flat_type_5 ROOM": 0.0, "flat_type_EXECUTIVE": 0.0, "flat_type_MULTI-GENERATION": 0.0, "storey_range_01 TO 03": 0.0, "storey_range_04 TO 06": 0.0, "storey_range_07 TO 09": 0.0, "storey_range_10 TO 12": 0.0, "storey_range_13 TO 15": 0.0, "storey_range_16 TO 18": 1.0, "storey_range_19 TO 21": 0.0, "storey_range_22 TO 24": 0.0, "storey_range_25 TO 27": 0.0, "storey_range_28 TO 30": 0.0, "storey_range_31 TO 33": 0.0, "storey_range_34 TO 36": 0.0, "storey_range_37 TO 39": 0.0, "storey_range_40 TO 42": 0.0, "storey_range_43 TO 45": 0.0, "storey_range_46 TO 48": 0.0, "storey_range_49 TO 51": 0.0, "flat_model_2-room": 0.0, "flat_model_3Gen": 0.0, "flat_model_Adjoined flat": 0.0, "flat_model_Apartment": 0.0, "flat_model_DBSS": 0.0, "flat_model_Improved": 0.0, "flat_model_Improved-Maisonette": 0.0, "flat_model_Maisonette": 0.0, "flat_model_Model A": 1.0, "flat_model_Model A-Maisonette": 0.0, "flat_model_Model A2": 0.0, "flat_model_Multi Generation": 0.0, "flat_model_New Generation": 0.0, "flat_model_Premium Apartment": 0.0, "flat_model_Premium Apartment Loft": 0.0, "flat_model_Simplified": 0.0, "flat_model_Standard": 0.0, "flat_model_Terrace": 0.0, "flat_model_Type S1": 0.0, "flat_model_Type S2": 0.0, "floor_area_sqm": -1.213303406442646, "month": -0.8295375485188797, "lease_commence_date": 1.3510964377576309, "remaining_lease": 1.324480614911619, "latitude": 1.4501003, "longitude": 103.8270738}' http://localhost:5001/predict
   ```

   You should receive a response with the predicted resale price.

#### 2. Running the Gradio Web UI

1. **Start the Gradio Interface**:
   To run the Gradio web UI, execute the following command:
   ```bash
   python3 api/gard.py
   ```
   The Gradio interface will start and can be accessed locally at `http://127.0.0.1:5155`.

## Folder Descriptions

### api

The `api` folder contains the necessary files and configurations for deploying and serving a machine learning model that predicts house prices in Singapore using Flask, Docker, and MLflow. This folder includes scripts to start a Flask application, a Dockerfile for containerizing the application, and a script for creating a web interface using Gradio.

#### Files:

1. **app.py:** Sets up a Flask web application to serve the machine learning model.
2. **Dockerfile:** Used to create a Docker image for the application.
3. **gard.py:** Creates a web interface using Gradio.

### configs

The `configs` folder contains configuration files used in the M

LOps project. These configuration files define various parameters and settings for different stages of the machine learning pipeline, including data preprocessing, feature engineering, and model training.

#### Files:

1. **config.yaml:** Contains configuration parameters for data preprocessing, feature engineering, and model training.

### data

The `data` folder holds the raw and processed data used in the project. This includes the original dataset, any intermediate data files, and the final dataset used for training and evaluating the machine learning model.

### data_base

The `data_base` folder serves as a backup storage for the dataset. This ensures that the original dataset is preserved and can be accessed if needed for any reason during the project lifecycle.

### docs

The `docs` folder contains project documentation. This includes the README file, the final project report, and any additional documentation generated during the project. The documentation provides an overview of the project, explains the methodology used, and presents the results and findings.

### models

The `models` folder stores the trained machine learning models. This includes the final model used for predictions as well as any intermediate models generated during the training process. The folder also contains model evaluation metrics and performance reports.

### notebooks

The `notebooks` folder contains Jupyter notebooks used for exploratory data analysis (EDA), model training, and evaluation. These notebooks provide a detailed view of the data analysis and modeling process and are useful for reproducibility and further exploration.

### outputs

The `outputs` folder holds the results generated by the machine learning model. This includes predictions, visualizations, and any other outputs generated during the project. The outputs are used to evaluate the model's performance and to present the final results.

### pipelines

The `pipelines` folder contains scripts and configurations for building and running the machine learning pipeline. The pipeline automates the data processing, feature engineering, model training, and evaluation steps. This folder ensures that the entire machine learning workflow can be executed in a reproducible and efficient manner.

### results

The `results` folder stores the final results of the project. This includes the model's performance metrics, visualizations, and any other outputs generated during the evaluation phase. The results are used to assess the success of the project and to communicate the findings to stakeholders.

### scripts

The `scripts` folder contains utility scripts used in the project. This includes scripts for data preprocessing, feature engineering, model training, and evaluation. The scripts automate various tasks in the machine learning pipeline, making the workflow more efficient and reproducible.

### services

The `services` folder contains files and configurations for deploying the machine learning model as a service. This includes scripts for setting up a web server, defining API endpoints, and integrating the model with a web interface. The services folder ensures that the model can be easily deployed and accessed by users.

### sql

The `sql` folder holds SQL scripts used for data extraction and manipulation. This includes scripts for querying the database, extracting relevant data, and performing any necessary transformations. The SQL scripts are used to prepare the data for analysis and modeling.

### src

The `src` folder contains the source code for the project. This includes the main codebase for data preprocessing, feature engineering, model training, and evaluation. The src folder ensures that all the code is organized and easily accessible.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
