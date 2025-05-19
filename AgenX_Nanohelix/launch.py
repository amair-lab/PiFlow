import logging
from flask import Flask, request, jsonify
import torch
import numpy as np
import pandas as pd
import joblib
import os

# Define the parameter ranges based on the physical constraints
PARAMETER_RANGES = {
    'pitch': {'min': 60, 'max': 200, 'unit': 'nm'},
    'fiber_radius': {'min': 20, 'max': 60, 'unit': 'nm'},
    'n_turns': {'min': 3, 'max': 10, 'unit': ''},
    'helix_radius': {'min': 20, 'max': 90, 'unit': 'nm'}
}


def validate_parameters(params_dict):
    is_valid = True
    error_msg = ""
    out_of_range_params = []

    for param_name, value in params_dict.items():
        if param_name in PARAMETER_RANGES:
            min_val = PARAMETER_RANGES[param_name]['min']
            max_val = PARAMETER_RANGES[param_name]['max']
            unit = PARAMETER_RANGES[param_name]['unit']

            # Calculate extended range with 10% tolerance
            extended_min = min_val * 0.9
            extended_max = max_val * 1.1

            if value < extended_min or value > extended_max:
                is_valid = False
                out_of_range_params.append({
                    'param': param_name,
                    'value': value,
                    'acceptable_range': f"{min_val}-{max_val} {unit}",
                    'extended_range': f"{extended_min:.1f}-{extended_max:.1f} {unit}"
                })

    if not is_valid:
        error_msg = "Input parameters outside acceptable range: "
        for param in out_of_range_params:
            error_msg += f"\n- {param['param']}: {param['value']} (acceptable range: {param['acceptable_range']}, with 10% tolerance: {param['extended_range']})"

    return is_valid, error_msg


def predict_g_factor(params_dict):
    # Extract parameters from dictionary
    pitch = params_dict['pitch']
    fiber_radius = params_dict['fiber_radius']
    n_turns = params_dict['n_turns']
    helix_radius = params_dict['helix_radius']

    # Check if model and scalers exist
    model_path = "models/nanohelix_mlp_model.pkl"
    scaler_X_path = "models/nanohelix_scaler_X.pkl"
    scaler_y_path = "models/nanohelix_scaler_y.pkl"

    if not all(os.path.exists(p) for p in [model_path, scaler_X_path, scaler_y_path]):
        raise FileNotFoundError("Model files not found. Please train the model first.")

    # Load model and scalers
    model = joblib.load(model_path)
    scaler_X = joblib.load(scaler_X_path)
    scaler_y = joblib.load(scaler_y_path)

    # Create a DataFrame with the 4 basic parameters
    data = pd.DataFrame({
        'pitch': [pitch],
        'fiber_radius': [fiber_radius],
        'n_turns': [n_turns],
        'helix_radius': [helix_radius]
    })

    # Compute derived parameters
    data_enriched = compute_nanohelix_parameters(data)

    # Get the expected feature names from the scaler
    expected_feature_names = scaler_X.feature_names_in_

    # Check if additional features are needed
    for feature in expected_feature_names:
        if feature not in data_enriched.columns:
            # Add default value of 0 for missing features
            data_enriched[feature] = 0

    # Ensure columns are in the same order as during training
    X = data_enriched[expected_feature_names]

    # Scale features
    X_scaled = scaler_X.transform(X)

    # Make prediction
    y_pred_scaled = model.predict(X_scaled)
    g_factor = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))[0][0]

    # Create result dictionary
    result_dict = {
        'g_factor': g_factor,
        **params_dict,  # Include input parameters
        **{k: v for k, v in data_enriched.iloc[0].to_dict().items() if k not in params_dict}  # Add derived parameters
    }

    return result_dict


def compute_nanohelix_parameters(df):
    # Create a copy to avoid modifying the original
    df_enriched = df.copy()

    # Calculate derived parameters using vectorized operations
    pitch = df_enriched['pitch']
    fiber_radius = df_enriched['fiber_radius']
    n_turns = df_enriched['n_turns']
    helix_radius = df_enriched['helix_radius']

    # Calculate turn length for all rows at once
    turn_length = np.sqrt((2 * np.pi * helix_radius) ** 2 + pitch ** 2)

    # Add all derived parameters using vectorized operations
    df_enriched['total_length'] = turn_length * n_turns
    df_enriched['height'] = pitch * n_turns
    df_enriched['curl'] = helix_radius / (helix_radius ** 2 + (pitch / (2 * np.pi)) ** 2)
    df_enriched['angle'] = np.arctan2(pitch, 2 * np.pi * helix_radius)
    df_enriched['total_fiber_length'] = df_enriched['total_length'] * (1 + (2 * np.pi * fiber_radius) / turn_length)
    df_enriched['V'] = np.pi * fiber_radius ** 2 * df_enriched['total_fiber_length']
    df_enriched['mass'] = df_enriched['V']

    return df_enriched


app = Flask(__name__)
app.logger.setLevel(logging.DEBUG)


@app.route('/predict', methods=['POST'])
def predict():
    submit = {
        "input": "",
        "output": "",
        "success": False,
        "error": ""
    }

    try:
        # Get input data from request
        data = request.get_json()
        app.logger.debug(f"Received input data: {data}")

        # Check if all required parameters are present
        required_params = ['pitch', 'fiber_radius', 'n_turns', 'helix_radius']
        missing_params = [param for param in required_params if param not in data]

        if missing_params:
            raise ValueError(f"Missing required parameters: {', '.join(missing_params)}")

        # Get prediction
        result = predict_g_factor(data)

        app.logger.debug(f"Predicted g-factor: {result['g_factor']:.2f}")

        # Update response with prediction
        submit["input"] = data
        submit["error"] = ""
        submit["output"] = result['g_factor']
        submit["success"] = True
        return jsonify(submit)

    except ValueError as e:
        app.logger.warning(f"Validation error: {str(e)}")
        submit["error"] = f"Validation error: {str(e)}"
        submit["input"] = data if 'data' in locals() else {}
        return jsonify(submit), 400  # Bad Request

    except Exception as e:
        app.logger.error(f"Error during prediction: {str(e)}")
        submit["error"] = f"Prediction failed: {str(e)}"
        return jsonify(submit), 500  # Internal Server Error


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=12501)