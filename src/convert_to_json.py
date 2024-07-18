import pandas as pd
import json

df = pd.read_csv('data/preprocessed/X.csv')

first_sample = df.iloc[0].to_dict()

first_sample.pop('Unnamed: 0', None)

# Convert the dictionary to a JSON string
first_sample_json = json.dumps(first_sample)



print(first_sample_json)