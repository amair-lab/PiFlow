import requests
import json

# Define the prediction endpoint
url = "http://127.0.0.1:12502/predict"

# Test cases with the correct data format based on the TSV sample
test_cases = [
    {"element": "Ba0.3La1.7Cu1O4-Y"}
]

# Make predictions for each test case
print("Testing Superconductor Tc Prediction API:")
print("-" * 50)

for i, test_case in enumerate(test_cases, 1):
    print(f"\nTest Case {i}: {test_case['element']}")

    # Make the POST request
    try:
        response = requests.post(
            url,
            data=json.dumps(test_case),
            headers={'Content-type': 'application/json'},
            timeout=10
        )

        # Check if request was successful
        if response.status_code == 200:
            result = response.json()
            if result["success"]:
                print(f"Predicted Tc: {result['output']:.2f} K")

                # Compare with expected values from the sample data
                expected_values = [29, 26, 19, 22, 23]
                if i <= len(expected_values):
                    print(f"Expected Tc: {expected_values[i - 1]} K")
                    print(f"Difference: {abs(result['output'] - expected_values[i - 1]):.2f} K")
            else:
                print(f"Error: {result['error']}")
        else:
            print(f"Request failed with status code: {response.status_code}")
            print(response.text)

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
