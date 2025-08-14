import pandas as pd
import json
import os

# Read the CSV file
csv_file = "../data/flab_metadata.csv"
df = pd.read_csv(csv_file)

# Function to safely split and clean key_words
def process_key_words(value):
    if isinstance(value, str):  # Check if value is a string
        return [keyword.strip() for keyword in value.split(",")]
    else:  # Handle missing or non-string values
        return []

# Process the data to create the "files" section
files_data = []
for _, row in df.iterrows():
    key_words_list = process_key_words(row['key_words'])
    path = f"../data/{row['category']}/{row['filename']}" if not pd.isna(row['category']) and not pd.isna(row['filename']) else "none"
    
    # Assert that the file exists if the path is not "none"
    if path != "none":
        assert os.path.isfile(path), f"File not found: {path}"

    file_entry = {
        "assay": row['assay/units'].split()[0] if isinstance(row['assay/units'], str) else "none",
        "category": row['category'] if not pd.isna(row['category']) else "none",
        "doi": row['doi'] if not pd.isna(row['doi']) else "none",
        "key_words": key_words_list if key_words_list else ["none"],
        "name": str(row['filename']) if not pd.isna(row['filename']) else "none",
        "path": path,
        "study": row['publication_title'] if not pd.isna(row['publication_title']) else "none",
        "year": int(row['year']) if not pd.isna(row['year']) else "none",
        "license": row['license'] if not pd.isna(row['license']) else "none",
        "size": int(row['size']) if not pd.isna(row['size']) else "none"
    }
    files_data.append(file_entry)

# Process the data to create the "keys" section
keys_data = {
    "assay": list(
        df['assay/units']
        .apply(lambda x: x.split()[0] if isinstance(x, str) else "none")
        .unique()
    ),
    "category": list(
        df['category']
        .apply(lambda x: x if not pd.isna(x) else "none")
        .unique()
    ),
    "doi": list(
        df['doi']
        .apply(lambda x: x if not pd.isna(x) else "none")
        .unique()
    ),
    "key_words": list(
        {keyword.strip()
         for keywords in df['key_words'].apply(lambda x: x if not pd.isna(x) else "")
         for keyword in process_key_words(keywords)} | {"none"}
    ),
    "study": list(
        df['publication_title']
        .apply(lambda x: x if not pd.isna(x) else "none")
        .unique()
    ),
    "year": sorted([
        int(year) if not pd.isna(year) else "none"
        for year in df['year'].unique()
    ]),
    "license": list(
        df['license']
        .apply(lambda x: x if not pd.isna(x) else "none")
        .unique()
    ),
    "size": [
        int(x) if not pd.isna(x) else "none"
        for x in df['size'].astype(object).unique()
    ]
}

# Create the final JSON structure
output_json = {
    "files": files_data,
    "keys": keys_data
}

# Save the JSON file
output_file = "../data/flab_metadata.json"
with open(output_file, 'w') as f:
    json.dump(output_json, f, indent=2)

print(f"JSON file saved as {output_file}")
