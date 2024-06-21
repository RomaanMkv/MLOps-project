import pandas as pd
import numpy as np

# Read the CSV file
file_path = 'data/flats.csv'
df = pd.read_csv(file_path)

# Shuffle the DataFrame rows
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Split the DataFrame into 5 equal parts
split_dfs = np.array_split(df, 5)

# Save each part into a separate CSV file
for i, split_df in enumerate(split_dfs):
    split_df.to_csv(f'sample{i+1}.csv', index=False)
