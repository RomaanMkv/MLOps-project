#!/bin/bash

# Loop to run the script 5 times with different example_version values
for version in {1..5}
do
    echo "Running prediction with example_version=$version"
    python src/predict.py example_version=$version
done
