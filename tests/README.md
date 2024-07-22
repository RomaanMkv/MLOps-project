# Data Folder

## Overview
The `data` folder contains all the data used in the MLOps project for predicting house prices in Singapore. This includes raw data, preprocessed data, samples, and scalers. Proper organization of this directory ensures that data handling is efficient and secure.

### coordinates.csv
This file contains additional features for the dataset, specifically full addresses and their corresponding latitude and longitude coordinates. These features are useful for geospatial analysis and enhancing the predictive power of the model.

#### First Few Lines of coordinates Data:
```
full_address,latitude,longitude
"SERANGOON, SERANGOON NTH AVE 2, block 139, Singapore",1.3646649,103.8722762
"WOODLANDS, MARSILING LANE, block 215, Singapore",1.4476136,103.7720555
"PASIR RIS, PASIR RIS DR 3, block 631, Singapore",1.3789831,103.9401988
"SENGKANG, RIVERVALE DR, block 187A, Singapore",1.393737,103.905399
"PUNGGOL, PUNGGOL DR, block 664B, Singapore",1.4000792,103.9178359
"YISHUN, YISHUN RING RD, block 363, Singapore",1.4289271,103.845575
```

### preprocessed
This directory contains the preprocessed data that is ready for model training and evaluation. The data is split into features (`X.csv`) and target (`y.csv`).

### samples
This directory contains sample data files used for testing and validation purposes. These samples help in validating the data pipeline and ensuring that the processing steps are correctly implemented.

#### Files:
- **sample.csv**: A sample data file containing a subset of the raw data.
- **sample.csv.dvc**: DVC file for tracking the sample data version.

### scalers
This directory contains the scaler files used for data normalization and standardization. Scalers are crucial for preprocessing steps to ensure that the data is in the correct format and scale for the model.


## Conclusion
The `data` folder is a critical part of the MLOps project, containing all necessary data files organized for efficient access and processing. This structure ensures that data is handled securely and consistently throughout the project lifecycle. Proper management of raw, preprocessed, sample data, and scalers ensures the reproducibility and reliability of the machine learning pipeline.