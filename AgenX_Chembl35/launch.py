#!/usr/bin/env python
import os
import logging
from flask import Flask, request, jsonify

from AgenX_Chembl35.inference import predict_smiles

# --- Flask App ---
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = app.logger

# --- Configuration ---
MODEL_PATH = os.path.join("models", "best_r2_model.pt")
HOST = '127.0.0.1'  # Allow external connections
PORT = 12500


@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint for pChEMBL prediction from SMILES.

    Expected JSON input:
    {
        "smiles": "CC(=O)Oc1ccccc1C(=O)O"
    }

    Returns:
    {
        "input": "CC(=O)Oc1ccccc1C(=O)O",
        "output": 5.23,
        "success": true,
        "error": ""
    }
    """
    response = {
        "input": "",
        "output": "",
        "success": False,
        "error": ""
    }

    try:
        # Get JSON data from request
        data = request.get_json()
        if not data:
            response["error"] = "No JSON data provided"
            return jsonify(response), 400

        # Extract SMILES
        smiles = data.get('smiles')
        logger.debug(f"Received SMILES: {smiles}")

        if not smiles:
            response["error"] = "SMILES string is missing in request body"
            return jsonify(response), 400

        # Make prediction
        predicted = predict_smiles(smiles, model_path=MODEL_PATH)
        logger.debug(f"Predicted value: {predicted}")

        if predicted is not None:
            response["input"] = smiles
            response["output"] = float(predicted)
            response["success"] = True
            return jsonify(response), 200
        else:
            response["input"] = smiles
            response["error"] = "Invalid SMILES string or prediction failed"
            return jsonify(response), 400

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        response["error"] = f"Internal server error: {str(e)}"
        return jsonify(response), 500


@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint to verify the service is running.
    """
    return jsonify({
        "status": "healthy",
        "model_path": MODEL_PATH,
        "success": True
    }), 200


@app.route('/', methods=['GET'])
def home():
    """
    Simple home page with API documentation.
    """
    return jsonify({
        "service": "MoleculeGCN pChEMBL Predictor",
        "version": "1.0",
        "endpoints": {
            "/": "API documentation (this page)",
            "/health": "Health check endpoint",
            "/predict": "POST endpoint for pChEMBL prediction"
        },
        "usage": {
            "method": "POST",
            "url": "/predict",
            "headers": {"Content-Type": "application/json"},
            "body": {"smiles": "your_smiles_string_here"}
        }
    }), 200


if __name__ == '__main__':
    # Verify model file exists
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model file not found at {MODEL_PATH}")
        exit(1)

    logger.info(f"Starting Flask server on {HOST}:{PORT}")
    logger.info(f"Using model from {MODEL_PATH}")

    app.run(
        debug=True,
        host=HOST,
        port=PORT
    )