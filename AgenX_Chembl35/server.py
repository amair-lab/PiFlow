#!/usr/bin/env python
import requests
import json

# API endpoint
BASE_URL = "http://127.0.0.1:12500"

# Test predictions with different SMILES
test_molecules = [
    'n1cccc2ccccc12'
]

for smiles in test_molecules:
    """Test the prediction endpoint."""
    headers = {'Content-Type': 'application/json'}
    data = {'smiles': smiles}

    response = requests.post(f"{BASE_URL}/predict", headers=headers, json=data)
    print(f"Prediction for SMILES '{smiles}':")
    print(json.dumps(response.json(), indent=2))
    print(f"Status Code: {response.status_code}\n")
