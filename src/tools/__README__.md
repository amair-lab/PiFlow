## Tools Protocol

> Tutorial of creating new tools.



### 🌲 Overview

```
.
├── README.md               # (you are here)
├── __init__.py
├── _chembl33_tools.py      # for task specific tools design. 
├── experiment_tools.py     # for experiment agent.
├── research_tools.py       # for research specific usage, e.g., searching and retrieving. 
├── tools_registry.py       # A FIXED FILE FOR LOADING @tool, DO NOT CHANGE. 
└── utils.py
```

### ⚠️ Guidance

If you want to introduce new tools, please go on these rules...

#### Rule 1: Every tool must contain at least these keys:

- tool_name: str
- success: bool
- error: None | str

#### Rule 2: Must have @tool assigned:

```
@tool(
    name="characterize_pchembl_value",
    description="Characterize the bioactivity of molecules by predicting their pChEMBL values using their SMILES representations and the ChEMBL database API. This tool leverages the ChEMBL API to estimate pChEMBL values, a crucial metric in drug discovery for quantifying compound potency.",
)
def characterize_pchembl_value(
    smiles: str,
) -> Dict[str, Any]:
...
```

With the @tool desc and name, the function should also have the type annotation with pydantic style. 


#### Rule 3: MUST have try ... except ... clause:

```
try:
    ...
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
```
