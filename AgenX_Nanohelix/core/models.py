import numpy as np
import pandas as pd
import joblib
import os


def predict_g_factor(params_dict):
    """
    Predict the g-factor for a nanohelix structure given a dictionary of the 4 basic parameters.

    Parameters:
    -----------
    params_dict : dict
        Dictionary containing the 4 basic parameters:
        - 'pitch': The pitch of the helix
        - 'fiber_radius': The radius of the fiber
        - 'n_turns': Number of turns in the helix
        - 'helix_radius': Radius of the helix

    Returns:
    --------
    result_dict : dict
        Dictionary containing:
        - 'g_factor': Predicted g-factor
        - All input parameters
        - All derived parameters
    """
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
