import torch
from src.model import TcPredictor
from src.data_processor import SuperconDataProcessor


class Inference:
    def __init__(self, model_path, input_size):
        self.model_path = model_path
        self.input_size = input_size
        self.device = torch.device('cpu')  # or 'cuda' if you want to run on GPU
        self.processor = SuperconDataProcessor()

        # Load model
        self.model = TcPredictor(input_size=self.input_size)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()
        self.model.to(self.device)
        print(f"Inference Model loaded from: {self.model_path}")

    def predict_tc_from_formula(self, chemical_formula, structure_type=None):
        """
        Predict the critical temperature (Tc) from a chemical formula.

        Args:
            chemical_formula (str): Chemical formula of the material
            structure_type (str, optional): Structure type code

        Returns:
            float: Predicted critical temperature in Kelvin
        """
        # Prepare input data
        input_data = {
            'element': chemical_formula
        }
        if structure_type:
            input_data['str3'] = structure_type

        # Process input
        try:
            processed_input = self.processor.process_input(input_data)
            input_tensor = torch.FloatTensor(processed_input).to(self.device)

            # Make prediction
            with torch.no_grad():
                prediction = self.model(input_tensor)

            return prediction.item()

        except Exception as e:
            print(f"Error during prediction: {e}")
            return None


if __name__ == "__main__":
    # Configuration
    MODEL_PATH = "./models/best_supercon_model.pth"
    INPUT_SIZE = 526  # Update with the actual feature dimension after training

    # Sample materials for testing
    sample_materials = [
        {"formula": "Ba0.2La1.8Cu1O4-Y", "structure": "T"},
        {"formula": "Ba0.1La1.9Ag0.1Cu0.9O4-Y", "structure": "T"},
    ]

    # Initialize inference model
    inference_model = Inference(
        model_path=MODEL_PATH,
        input_size=INPUT_SIZE
    )

    # Perform predictions
    print("\n--- Critical Temperature (Tc) Predictions ---")
    for material in sample_materials:
        predicted_tc = inference_model.predict_tc_from_formula(
            material["formula"],
            material["structure"]
        )
        if predicted_tc is not None:
            print(f"Formula: {material['formula']:<30} Predicted Tc: {predicted_tc:.2f} K")