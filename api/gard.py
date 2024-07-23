import gradio as gr
from utils import init_hydra
from data import transform_data
import json
import requests
import pandas as pd
import re
from datetime import datetime

port_number = 5001

cfg = init_hydra()

current_year = datetime.now().year
years = list(range(current_year - 100, current_year + 1))
months = [f"{i:02d}" for i in range(0, 12)]

# Define the predict function
def predict(month=None,
            town=None,
            flat_type=None,
            block=None,
            street_name=None,
            storey_range=None,
            floor_area_sqm=None,
            flat_model=None,
            lease_commence_date=None,
            remaining_lease_years=None,
            remaining_lease_months=None):

    # Combine years and months into the desired format
    remaining_lease = f"{int(remaining_lease_years)} years {int(remaining_lease_months):02d} months"
    
    # Create features dictionary
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
    
    # Build a dataframe of one row
    raw_df = pd.DataFrame(features, index=[0])
    
    # Transform the input data
    X = transform_data(df=raw_df, cfg=cfg)
    
    # Convert to JSON
    example = X.iloc[0,:]
    example = json.dumps(example.to_dict())

    payload = example

    # Send POST request with the payload to the deployed Model API
    response = requests.post(
        url=f"http://localhost:{port_number}/predict",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    
    return response.json()

def validate_month_format(month):
    pattern = re.compile(r'^\d{4}-\d{2}$')
    if pattern.match(month):
        return True, ""
    else:
        return False, "Invalid format. Please enter the date in YYYY-MM format."

# Define Gradio interface
demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(label="Month", placeholder="YYYY-MM", validate=validate_month_format), 
        gr.Dropdown(label="Town", choices=list(cfg.town_choices)),
        gr.Dropdown(label="Flat Type", choices=list(cfg.flat_type_choices)),
        gr.Dropdown(label="Block", choices=list(cfg.block_choices)),
        gr.Dropdown(label="Street Name", choices=list(cfg.street_name_choices)),   
        gr.Dropdown(label="Storey Range", choices=list(cfg.storey_range_choices)), 
        gr.Number(label="Floor Area (sqm)"),
        gr.Dropdown(label="Flat Model", choices=list(cfg.flat_model_choices)),  
        gr.Dropdown(choices=years, label="Lease Commence Date"),
        gr.Number(label="Remaining Lease Years", precision=0),
        gr.Dropdown(label="Remaining Lease Months", choices=months),
    ],
    outputs=gr.Text(label="Prediction Result"),
)

if __name__ == "__main__":
    demo.launch(server_port=5155)
