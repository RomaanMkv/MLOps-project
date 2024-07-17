import gradio as gr
from utils import init_hydra
from data import transform_data
import json
import requests
import pandas as pd

port_number = 5001

cfg = init_hydra()

# You need to define a parameter for each column in your raw dataset
def predict(month = None,
            town = None,
            flat_type = None,
            block = None,
            street_name = None,
            storey_range = None,
            floor_area_sqm = None,
            flat_model = None,
            lease_commence_date = None,
            remaining_lease = None,
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
        example.to_dict()
    )

    payload = example

    # Send POST request with the payload to the deployed Model API
    # Here you can pass the port number at runtime using Hydra
    response = requests.post(
        url=f"http://localhost:{port_number}/predict",
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
    fn=predict,
    
    # Here, the arguments in `predict` function
    # will populated from the values of these input components
    inputs = [
        # Select proper components for data types of the columns in your raw dataset
        gr.Text(label = "month"), 
        gr.Dropdown(label="town", choices=list(cfg.town_choices)),
        gr.Dropdown(label="flat_type", choices=list(cfg.flat_type_choices)),
        gr.Dropdown(label="block", choices=list(cfg.block_choices)),
        gr.Dropdown(label="street_name", choices=list(cfg.street_name_choices)),   
        gr.Dropdown(label = "storey_range", choices=list(cfg.storey_range_choices)), 
        gr.Number(label="floor_area_sqm"),
        gr.Dropdown(label="flat_model", choices=list(cfg.flat_model_choices)),  
        gr.Number(label="lease_commence_date"),
        gr.Text(label="remaining_lease"),
    ],
    
    # The outputs here will get the returned value from `predict` function
    outputs = gr.Text(label="prediction result"),
    
    # This will provide the user with examples to test the API
    # examples="data/examples"
    # data/examples is a folder contains a file `log.csv` 
    # which contains data samples as examples to enter by user 
    # when needed. 
)

if __name__ == "__main__":
    # Launch the web UI locally on port 5155
    demo.launch(server_port = 5155)

    # Launch the web UI in Gradio cloud on port 5155
    # demo.launch(share=True, server_port = 5155)