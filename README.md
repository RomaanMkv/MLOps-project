# README: HDB Resale Price Prediction Project

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

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
