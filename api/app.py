from flask import Flask, request, jsonify, make_response
import mlflow
import mlflow.pyfunc
import os
import pandas as pd

import gradio as gr
import mlflow
from utils import init_hydra
from model import load_features
from data import transform_data
import json
import requests
import numpy as np
import pandas as pd

cfg = init_hydra()
port_number = 5001

BASE_PATH = os.path.expandvars("$PROJECT_BASE_PATH")

# Load the model from the specified directory
model = mlflow.pyfunc.load_model(os.path.join(BASE_PATH, "api", "model_dir"))

app = Flask(__name__)

@app.route("/info", methods=["GET"])
def info():
    # Get model metadata
    metadata = model.metadata.to_dict()
    response = make_response(jsonify(metadata), 200)
    response.content_type = "application/json"
    return response

@app.route("/", methods=["GET"])
def home():
    msg = """
    Welcome to our ML service to predict Customer satisfaction\n\n

    This API has two main endpoints:\n
    1. /info: to get info about the deployed model.\n
    2. /predict: to send predict requests to our deployed model.\n
    """

    response = make_response(msg, 200)
    response.content_type = "text/plain"
    return response

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get the JSON data from the request
        input_data = request.get_json()

        # Ensure the input data is valid
        if input_data is None:
            raise ValueError("No input data provided")

        # Convert the input data into the format expected by the model
        input_df = pd.DataFrame([input_data])

        # Make the prediction using the loaded model
        prediction = model.predict(input_df)

        # Return the prediction result
        return jsonify({'result': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    

# You need to define a parameter for each column in your raw dataset
def predict1(month = None,
            town = None,
            flat_type = None,
            block = None,
            street_name = None,
            storey_range = None,
            floor_area_sqm = None,
            flat_model = None,
            lease_commence_date = None,
            remaining_lease = None,
            resale_price = None,
            ):
    
    # This will be a dict of column values for input data sample
    features = {
        "month": month,
        "town": town,
        "flat_type": flat_type,
        "block": block,
        "street_name": street_name,
        "storey_range": storey_range,
        "floor_area_sqm": floor_area_sqm,
        "flat_model": flat_model,
        "lease_commence_date": lease_commence_date,
        "remaining_lease": remaining_lease,
        "resale_price": resale_price
    }
    
    # print(features)
    
    # Build a dataframe of one row
    raw_df = pd.DataFrame(features, index=[0])
    
    # This will read the saved transformers "v4" from ZenML artifact store
    # And only transform the input data (no fit here).
    X = transform_data(
                        df = raw_df, 
                        cfg = cfg
                      )
    
    # Convert it into JSON
    example = X.iloc[0,:]

    example = json.dumps( 
        { "inputs": example.to_dict() }
    )

    payload = example

    # Send POST request with the payload to the deployed Model API
    # Here you can pass the port number at runtime using Hydra
    response = requests.post(
        url=f"http://localhost:{port_number}/invocations",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    
    # Change this to some meaningful output for your model
    # For classification, it returns the predicted label
    # For regression, it returns the predicted value
    return response.json()

# Only one interface is enough
demo = gr.Interface(
    # The predict function will accept inputs as arguments and return output
    fn=predict1,
    
    # Here, the arguments in `predict` function
    # will populated from the values of these input components
    inputs = [
        # Select proper components for data types of the columns in your raw dataset
        gr.Number(label = "age"), 
        gr.Text(label="job"),
        gr.Text(label="marital"),
        gr.Text(label="education"),
        gr.Dropdown(label="default", choices=["no", "yes"]),   
        gr.Number(label = "balance"), 
        gr.Dropdown(label="housing", choices=["no", "yes"]),   
        gr.Dropdown(label="loan", choices=["no", "yes"]),   
        gr.Text(label="contact"),
        gr.Text(label="day_of_week"),
        gr.Text(label="month"),
        gr.Number(label = "duration"), 
        gr.Number(label = "campaign"), 
        gr.Number(label = "pdays"), 
        gr.Number(label = "previous"),
        gr.Text(label="poutcome"),
    ],
    
    # The outputs here will get the returned value from `predict` function
    outputs = gr.Text(label="prediction result"),
    
    # This will provide the user with examples to test the API
    examples="data/examples"
    # data/examples is a folder contains a file `log.csv` 
    # which contains data samples as examples to enter by user 
    # when needed. 
)

# This will run a local server to accept requests to the API.
if __name__ == "__main__":
    port = int(os.environ.get('PORT', port_number))
    app.run(debug=True, host='0.0.0.0', port=port)
    
    # Launch the web UI locally on port 5155
    demo.launch(server_port = 5155)

    # Launch the web UI in Gradio cloud on port 5155
    # demo.launch(share=True, server_port = 5155)
