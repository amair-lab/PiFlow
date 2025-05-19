import json
import requests
from typing import Dict, Any, List, Optional, Union, Annotated

from src.tools.tools_registry import tool


@tool(
    name="characterize_pchembl_value",
    description="Characterize the bioactivity of molecules by predicting their pChEMBL values using their SMILES representations and the ChEMBL database API. This tool leverages the ChEMBL API to estimate pChEMBL values, a crucial metric in drug discovery for quantifying compound potency.",
)
def characterize_pchembl_value(
    smiles: str,
) -> Dict[str, Any]:
    """
    Submit SMILES representations of molecules to the chembl33 API to predict their
    pchembl_value_mean_BF values.

    The pChEMBL value is a key descriptor in drug discovery representing the negative
    log-transformed potency value of a compound's biological activity (e.g., Ki, Kd, IC50, EC50).
    Higher values indicate stronger biological activity.

    Args:
        smiles: Either a single SMILES string or a list of SMILES strings representing molecules

    Returns:
        Dictionary containing prediction results for each molecule
    """
    # Convert single SMILES to list for consistent handling
    if not isinstance(smiles, str):
        return {
            "tool_name": "characterize_pchembl_value",
            "success": False,
            "error": f"Must be only ONE smiles string.",
            "smiles": smiles
        }

    # Construct API URL
    try:
        response = requests.post(
            "http://127.0.0.1:12500/predict",
            data=json.dumps({'smiles': smiles}),
            headers={'Content-type': 'application/json'}
        ).json()

        response["tool_name"] = "characterize_pchembl_value"
        if response["success"]:
            response["success"] = True
        return response

    except requests.exceptions.RequestException as e:
        return {
            "tool_name": "characterize_pchembl_value",
            "success": False,
            "error": f"API request failed: {str(e)}",
            "smiles": smiles
        }
    except Exception as e:
        return {
            "tool_name": "characterize_pchembl_value",
            "success": False,
            "error": f"Error processing request: {str(e)}",
            "smiles": smiles
        }


if __name__ == '__main__':

    # test tools.
    result_characterize_pchembl_value = characterize_pchembl_value("CCOCC")
    print("result_characterize_pchembl_value: ")
    print(result_characterize_pchembl_value, end="\n\n")