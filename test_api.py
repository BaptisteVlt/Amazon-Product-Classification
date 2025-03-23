import requests
import pandas as pd
import random
import json

# Load the dataset
file_path = "amz_products_small.jsonl"

# Read all lines from the JSONL file
with open(file_path, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

# Randomly select 10 rows
random_samples = random.sample(data, 10)

# API endpoint
API_URL = "http://127.0.0.1:8000/predict"

# Store results
results = []

# Send requests to API
for sample in random_samples:
    description = sample.get("description", "")
    true_label = sample.get("main_cat", "Unknown")

    if not description:
        continue  # Skip empty descriptions

    payload = {"description": description}
    response = requests.post(API_URL, json=payload)

    if response.status_code == 200:
        predicted_label = response.json().get("main_cat", "Error")
    else:
        predicted_label = "Error"

    # Store result
    results.append({
        "True Label": true_label,
        "Predicted Label": predicted_label,
        "Match": "✅" if true_label == predicted_label else "❌"
    })

# Convert results to DataFrame and display
df_results = pd.DataFrame(results)
print(df_results)
