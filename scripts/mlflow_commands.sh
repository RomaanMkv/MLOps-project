# extracts data
mlflow run . -e extract --env-manager=local
# transforms data
mlflow run . -e transform --env-manager=local
# run model creation
mlflow run . --env-manager=local
# evaluate model
mlflow run . -e evaluate --env-manager=local
# validate all models
mlflow run . -e validate --env-manager=local
# predict on deployed model
mlflow run . -e predict --env-manager=local
# run mlflow server
mlflow server
