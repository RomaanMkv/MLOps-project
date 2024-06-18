#!/bin/bash

# Navigate to the script's directory
cd "$(dirname "$0")"

# Navigate to the project root directory
cd ..

# Check if virtual environment folder exists, if not, create one
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Activate the virtual environment
source venv/bin/activate

# Install the requirements
pip install -r requirements.txt

# Deactivate the virtual environment
deactivate
