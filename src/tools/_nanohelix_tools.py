import json
import requests
from typing import Dict, Any, List, Optional, Union, Annotated

from src.tools.tools_registry import tool


@tool(
    name="characterize_nanohelix_gfactor",
    description="Characterize the g-factor (dissymmetry factor) of nanohelices based on their geometric parameters. The input values MUST be 4 values. They are fiber_radius, helix_radius, n_turns and pitch. The `fiber_radius` is the actual fiber/wire that forms the helix (in nm). The `helix_radius` is the distance from central axis to the center of the helical path (in nm). The `n_turns` is the number of turns in the helix. The `pitch` is axial distance between adjacent turns (in nm). All input are float without units such as 'nm' or 'turns'. No units are required. ",
)
def characterize_nanohelix_gfactor(
        fiber_radius: Annotated[float, "Radius of the actual fiber/wire that forms the helix (in nm)"],
        helix_radius: Annotated[float, "Radius of the helix - distance from central axis to the center of the helical path (in nm)"],
        n_turns: Annotated[float, "Number of complete turns in the helix"],
        pitch: Annotated[float, "Axial distance between adjacent turns (in nm)"]
) -> Dict[str, Any]:
    """
    Predicts the g-factor (dissymmetry factor) of a nanohelix based on its geometric parameters.

    The g-factor quantifies the different optical responses of nanohelices to left and right
    circularly polarized light. Higher g-factor values indicate stronger chiral optical activity.

    Args:
        fiber_radius: Radius of the actual fiber/wire that forms the helix (in nm)
        helix_radius: Radius of the helix - distance from central axis to the center of the helical path (in nm)
        n_turns: Number of complete turns in the helix
        pitch: Axial distance between adjacent turns (in nm)

    Returns:
        Dictionary containing the prediction result with the g-factor value
    """
    # Construct input data
    input_data = {
        "fiber_radius": fiber_radius,
        "helix_radius": helix_radius,
        "n_turns": n_turns,
        "pitch": pitch
    }

    # Call the API
    try:
        response = requests.post(
            "http://127.0.0.1:12501/predict",
            data=json.dumps(input_data),
            headers={'Content-type': 'application/json'}
        ).json()

        # Format the response to match the required structure
        return {
            "tool_name": "predict_nanohelix_gfactor",
            "input": input_data,
            "output": response["output"],
            "success": response["success"],
            "error": response["error"] if not response["success"] else None
        }

    except requests.exceptions.RequestException as e:
        return {
            "tool_name": "predict_nanohelix_gfactor",
            "input": input_data,
            "output": None,
            "success": False,
            "error": f"API request failed: {str(e)}"
        }
    except Exception as e:
        return {
            "tool_name": "predict_nanohelix_gfactor",
            "input": input_data,
            "output": None,
            "success": False,
            "error": f"Error processing request: {str(e)}"
        }


if __name__ == '__main__':
    # Test predict_nanohelix_gfactor tool
    result_predict = characterize_nanohelix_gfactor(
        fiber_radius=0.4,
        helix_radius=0.4,
        n_turns=9,
        pitch=0.74
    )
    print("predict_nanohelix_gfactor result:")
    print(result_predict, end="\n\n")