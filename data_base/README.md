# data_base Folder - Detailed README

## Overview
The `data_base` folder is considered a static data source for the MLOps project predicting house prices in Singapore. This folder contains several sample datasets that are subsets of the entire dataset. Each sample represents 20% of the total rows in the full dataset. Versioning is managed using DVC to ensure no data samples are lost and to maintain data integrity across different versions.

## File Structure
The `data_base` folder is organized as follows:

```
.
├── README.md
├── sample1.csv
├── sample2.csv
├── sample3.csv
├── sample4.csv
└── sample5.csv
```

### README.md
This file provides an overview and description of the contents and purpose of the `data_base` folder.

### Sample Data Files
Each sample data file (`sample1.csv`, `sample2.csv`, `sample3.csv`, `sample4.csv`, `sample5.csv`) contains a subset of the full dataset. These files are used for various stages of the machine learning pipeline, including testing, validation, and model training. The sample size is 20% of the total rows in the entire dataset.

#### Example of Data Sample:
```
month,town,flat_type,block,street_name,storey_range,floor_area_sqm,flat_model,lease_commence_date,remaining_lease,resale_price
2019-03,SEMBAWANG,3 ROOM,590B,MONTREAL LINK,16 TO 18,68.0,Model A,2015,95 years 01 month,292000.0
```

### Versioning with DVC
To ensure that no data samples are lost and to maintain data integrity, versioning of the data is managed using Data Version Control (DVC). This allows for tracking changes, reproducing experiments, and sharing data and models. 

## Conclusion
The `data_base` folder serves as a static data source, providing essential subsets of the full dataset for various stages of the machine learning pipeline. Proper management and versioning of these data samples using DVC ensure the reproducibility and reliability of the project, preventing data loss and maintaining the integrity of the dataset across different versions.