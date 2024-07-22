# Results Folder 

## Overview
The `results` folder contains the performance evaluation results for various machine learning models. Each directory within this folder corresponds to a specific run of a model, with performance metrics visualized in PNG format. These results are critical for analyzing and comparing the performance of different models and configurations.

## Directory Structure
The `results` folder is organized into subdirectories, each representing a different run of a machine learning model. The structure is as follows:

```
.
├── child_multi_run_gradient_boosting_neg_mean_squared_error_-985232745_1586647_0
│   └── performance_plot.png
├── child_multi_run_gradient_boosting_neg_mean_squared_error_-985232745_1586647_1
│   └── performance_plot.png
...
├── child_multi_run_random_forest_neg_mean_squared_error_-2494654304_8045254_9
│   └── performance_plot.png
```

### Subdirectory Naming
- **Prefix**: `child_multi_run_` indicates that these are results from multiple runs of a model.
- **Model Type**: `gradient_boosting` or `random_forest` specifies the type of model used.
- **Metric**: `neg_mean_squared_error` denotes the evaluation metric used.
- **Unique Identifier**: A combination of numerical values serves as a unique identifier for each run.
- **Subdirectory Example**: `child_multi_run_gradient_boosting_neg_mean_squared_error_-985232745_1586647_0`

### Performance Plot (PNG)
- **File Name**: `performance_plot.png`
- **Description**: Each PNG file contains a plot illustrating the performance of the model for the specific run. This plot includes metrics such as negative mean squared error (Neg MSE).

