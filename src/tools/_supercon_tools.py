import json
import requests
from typing import Dict, Any, List, Optional, Union, Annotated

from src.tools.tools_registry import tool


@tool(
    name="characterize_Tc_value",
    description="Characterize the Tc value of a superconductor material. The input `element` is Chemical formula of the material (e.g., 'Ba0.2La1.8Cu1O4-Y'). The tool returns the predicted critical temperature (Tc) in Kelvin. Superconductors are materials that conduct electricity with zero resistance below a critical temperature (Tc). Higher Tc values are desirable for practical applications, as they require less cooling. Finding new superconductors with higher Tc values is a major goal in materials science. ",
)
def characterize_Tc_value(
    element: str,
) -> Dict[str, Any]:
    # Convert single element to list for consistent handling
    if not isinstance(element, str):
        return {
            "tool_name": "characterize_Tc_value",
            "success": False,
            "error": f"Must be only ONE element string of the material.",
            "element": element
        }

    # Construct API URL
    try:
        response = requests.post(
            "http://127.0.0.1:12502/predict",
            data=json.dumps({'element': element}),
            headers={'Content-type': 'application/json'}
        ).json()

        response["tool_name"] = "characterize_Tc_value"
        if response["success"]:
            response["success"] = True
        return response

    except requests.exceptions.RequestException as e:
        return {
            "tool_name": "characterize_Tc_value",
            "success": False,
            "error": f"API request failed: {str(e)}",
            "element": element
        }
    except Exception as e:
        return {
            "tool_name": "characterize_Tc_value",
            "success": False,
            "error": f"Error processing request: {str(e)}",
            "element": element
        }


if __name__ == '__main__':
    # test tools.
    result_characterize_pchembl_value = characterize_Tc_value(
        "Nb3Sn1"
    )

    print("result_characterize_pchembl_value: ")
    print(result_characterize_pchembl_value, end="\n\n")