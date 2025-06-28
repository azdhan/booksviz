import pandas as pd
import numpy as np
import json
import os

# Define file paths
input_csv_xz = 'GoodReads_100k_books.csv.xz'
output_json = 'scatter_data.json'

# Columns to read and their new names
# Corrected 'description' to 'desc' to match the CSV header
use_cols_map = {'pages': 'pages', 'desc': 'desc_col', 'reviews': 'reviews', 'rating': 'rating'} # Temporarily rename 'desc' to avoid clash
selected_cols_from_csv = list(use_cols_map.keys())

# Read the CSV file directly from the XZ archive
df = pd.read_csv(input_csv_xz, usecols=selected_cols_from_csv, compression='xz')

# Rename columns for internal use, especially 'desc' to 'desc_col'
df.rename(columns=use_cols_map, inplace=True)

# Compute blurb length from the renamed 'desc_col'
df['blurb'] = df['desc_col'].astype(str).str.len()
# Now drop the original description column, which is now 'desc_col'
df.drop(columns=['desc_col'], inplace=True)

# Identify numeric columns for outlier clipping
numeric_cols = ['pages', 'reviews', 'rating', 'blurb']

# Apply outlier clipping
for col in numeric_cols:
    # Convert to numeric, coercing errors to NaN. This is important for 'pages' which might have non-numeric values.
    df[col] = pd.to_numeric(df[col], errors='coerce')
    # Drop rows with NaN values that were created by coercion or were already present
    df.dropna(subset=[col], inplace=True)

    lower_percentile = df[col].quantile(0.005)
    upper_percentile = df[col].quantile(0.995)
    df = df[(df[col] >= lower_percentile) & (df[col] <= upper_percentile)]

# Randomly sample max 5000 rows
if len(df) > 5000:
    df = df.sample(n=5000, random_state=42)

# Prepare data for JSON output
output_data = df[['pages', 'blurb', 'reviews', 'rating']].to_dict(orient='records')

# Dump to JSON
with open(output_json, 'w') as f:
    json.dump(output_data, f)

# Check JSON file size
file_size_bytes = os.path.getsize(output_json)
print(f"Final JSON file size: {file_size_bytes} bytes")

if file_size_bytes > 2 * 1024 * 1024:
    print(f"Warning: File size ({file_size_bytes} bytes) exceeds 2MB limit.")

print(f"Successfully created {output_json} with {len(df)} rows.")
