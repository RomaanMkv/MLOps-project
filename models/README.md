# models Folder

## Overview
The `models` folder contains the machine learning models used in the project, along with their associated files. This directory is essential for versioning, managing, and sharing models within the team. The structure is organized to separate different models and their respective versions, ensuring clarity and ease of access.

## File Structure
The `models` folder is organized as follows:

```
.
├── gradient_boosting
│   ├── champion
│   │   ├── conda.yaml
│   │   ├── input_example.json
│   │   ├── MLmodel
│   │   ├── model.pkl
│   │   ├── python_env.yaml
│   │   ├── registered_model_meta
│   │   └── requirements.txt
│   └── validation_champion
│       ├── conda.yaml
│       ├── input_example.json
│       ├── MLmodel
│       ├── model.pkl
│       ├── python_env.yaml
│       ├── registered_model_meta
│       └── requirements.txt
├── random_forest
│   └── champion
│       ├── conda.yaml
│       ├── input_example.json
│       ├── MLmodel
│       ├── model.pkl
│       ├── python_env.yaml
│       ├── registered_model_meta
│       └── requirements.txt
└── README.md
```

### gradient_boosting
This subdirectory contains the Gradient Boosting models and their related files. It is further divided into different versions or validation states.

#### champion
This folder contains the champion version of the Gradient Boosting model. The champion model is considered the best performing model and is typically used in production.

#### Files:
- **conda.yaml**: Configuration file for creating a conda environment specific to the model.
- **input_example.json**: A JSON file containing an example of the input data expected by the model.
- **MLmodel**: The MLflow model configuration file that defines the model's metadata and dependencies.
- **model.pkl**: The serialized model file.
- **python_env.yaml**: Configuration file for setting up the Python environment.
- **registered_model_meta**: Metadata about the registered model.
- **requirements.txt**: A list of Python dependencies required by the model.

#### validation_champion
This folder contains the validation champion version of the Gradient Boosting model. This version is used for validation purposes and to ensure that the model performs well on unseen data.

#### Files:
- **conda.yaml**: Configuration file for creating a conda environment specific to the model.
- **input_example.json**: A JSON file containing an example of the input data expected by the model.
- **MLmodel**: The MLflow model configuration file that defines the model's metadata and dependencies.
- **model.pkl**: The serialized model file.
- **python_env.yaml**: Configuration file for setting up the Python environment.
- **registered_model_meta**: Metadata about the registered model.
- **requirements.txt**: A list of Python dependencies required by the model.

### random_forest
This subdirectory contains the Random Forest model and its related files. It is organized similarly to the `gradient_boosting` directory.

#### champion
This folder contains the champion version of the Random Forest model. The champion model is considered the best performing model and is typically used in production.

#### Files:
- **conda.yaml**: Configuration file for creating a conda environment specific to the model.
- **input_example.json**: A JSON file containing an example of the input data expected by the model.
- **MLmodel**: The MLflow model configuration file that defines the model's metadata and dependencies.
- **model.pkl**: The serialized model file.
- **python_env.yaml**: Configuration file for setting up the Python environment.
- **registered_model_meta**: Metadata about the registered model.
- **requirements.txt**: A list of Python dependencies required by the model.

## Conclusion
The `models` folder is a crucial part of the MLOps project, providing a structured and organized way to store, manage, and share machine learning models. By separating different models and their respective versions, the project ensures clarity and ease of access, facilitating better collaboration and model management.