#!/usr/bin/env python
import torch
import os
from typing import Dict, Tuple, List, Optional, Union

from src.model import MoleculeGCN, MoleculeGraph, Standardizer


class MoleculePredictor:
    """
    A simplified predictor class for MoleculeGCN inference.
    """

    def __init__(self, model_path: str = 'models/best_model.pt', device: Optional[str] = None):
        """
        Initialize the predictor.

        Args:
            model_path: Path to the trained model checkpoint
            device: Device to run inference on ('cuda', 'cpu', or None for auto-detect)
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model, self.standardizer, self.config = self._load_model(model_path)

    def _load_model(self, model_path: str) -> Tuple[MoleculeGCN, Standardizer, Dict]:
        """Load the model, standardizer, and configuration."""
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        config = checkpoint['config']

        model = MoleculeGCN(
            node_vec_len=config['model']['node_vec_len'],
            node_fea_len=config['model']['node_fea_len'],
            hidden_fea_len=config['model']['hidden_fea_len'],
            n_conv=config['model']['n_conv'],
            n_hidden=config['model']['n_hidden'],
            n_outputs=config['model']['n_outputs'],
            p_dropout=config['model']['p_dropout'],
        )

        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()

        standardizer = Standardizer()
        standardizer.load_state_dict(checkpoint['standardizer'])

        return model, standardizer, config

    def predict(self, smiles: Union[str, List[str]]) -> Union[float, List[float], None]:
        """
        Predict pChEMBL values for SMILES string(s).

        Args:
            smiles: A SMILES string or list of SMILES strings

        Returns:
            float or list of floats: Predicted pChEMBL value(s)
            None: If the SMILES is invalid
        """
        if isinstance(smiles, str):
            return self._predict_single(smiles)
        else:
            return [self._predict_single(s) for s in smiles]

    def _predict_single(self, smiles: str) -> Optional[float]:
        """Predict pChEMBL value for a single SMILES string."""
        try:
            graph = MoleculeGraph(
                smiles,
                node_vec_len=self.config['model']['node_vec_len'],
                max_atoms=self.config['data']['max_atoms']
            )

            # Check if molecule is valid
            if not hasattr(graph, 'node_mat') or not hasattr(graph, 'adj_mat'):
                print(f"Invalid SMILES: {smiles}")
                return None

            node_mat = torch.FloatTensor(graph.node_mat).unsqueeze(0).to(self.device)
            adj_mat = torch.FloatTensor(graph.adj_mat).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(node_mat, adj_mat)
                prediction = self.standardizer.unstandardize(output).item()

            return prediction

        except Exception as e:
            print(f"Error processing SMILES '{smiles}': {str(e)}")
            return None


# Simple function for direct testing
def predict_smiles(smiles: Union[str, List[str]],
                   model_path: str = 'models/best_r2_model.pt',
                   device: Optional[str] = None) -> Union[float, List[float], None]:
    """
    Direct function to predict pChEMBL values from SMILES strings.

    Args:
        smiles: A SMILES string or list of SMILES strings
        model_path: Path to the trained model checkpoint
        device: Device to run inference on ('cuda', 'cpu', or None for auto-detect)

    Returns:
        float or list of floats: Predicted pChEMBL value(s)
        None: If the SMILES is invalid
    """
    predictor = MoleculePredictor(model_path, device)
    return predictor.predict(smiles)


# Example usage
if __name__ == "__main__":
    # Test with single SMILES
    smiles = 'CC(=O)Oc1ccccc1C(=O)O'  # Aspirin
    prediction = predict_smiles(smiles, model_path='models/best_r2_model.pt')
    print(f"SMILES: {smiles}")
    print(f"Prediction: {prediction}")
