import os
import yaml
import json
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Dictionary containing the configuration
    """
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading YAML config from {config_path}: {e}")
        return {}


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a file (supports YAML and JSON).

    Args:
        config_path: Path to the configuration file

    Returns:
        Dictionary containing the configuration
    """
    if config_path.endswith('.yaml') or config_path.endswith('.yml'):
        return load_yaml_config(config_path)
    elif config_path.endswith('.json'):
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading JSON config from {config_path}: {e}")
            return {}
    else:
        logger.error(f"Unsupported config file format: {config_path}")
        return {}


def get_agent_llm_config(agent_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get LLM configuration for a specific agent from its config.

    Args:
        agent_config: Agent configuration dictionary

    Returns:
        Dictionary with LLM configuration for the agent
    """
    if not agent_config or not agent_config.get('api_config'):
        return {}

    api_config = agent_config['api_config']

    # Get API key from environment variable if specified
    api_key = os.environ.get(api_config.get('api_key_env', ''), api_config.get('api_key', ''))

    # Build config_list entry
    config_entry = {
        "model": api_config.get('model_name', 'gpt-4'),
        "api_key": api_key
    }

    # Add base_url if specified
    if api_config.get('base_url'):
        config_entry["base_url"] = api_config.get('base_url')

    # Add any additional parameters from api_config
    for key, value in api_config.items():
        if key not in ['model_name', 'api_key', 'api_key_env', 'base_url']:
            config_entry[key] = value

    return {
        "model": api_config.get('model_name', 'gpt-4'),
        "temperature": api_config.get('temperature', 0.7),
        "config_list": [config_entry]
    }


def get_default_llm_config(model_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Get default LLM configuration.

    Args:
        model_name: Optional model name to use (defaults to environment variable or 'gpt-4')

    Returns:
        Dictionary with LLM configuration
    """
    model = model_name or os.environ.get("DEFAULT_LLM_MODEL", "gpt-4")

    return {
        "model": model,
        "temperature": 0.7,
        "config_list": [{"model": model, "api_key": os.environ.get("OPENAI_API_KEY", "")}]
    }


def get_tool_by_name(tool_name: str, available_tools: Dict[str, Any]) -> Optional[Any]:
    """
    Get a tool function by its name.

    Args:
        tool_name: Name of the tool to retrieve
        available_tools: Dictionary of available tools

    Returns:
        The tool function or None if not found
    """
    return available_tools.get(tool_name)


def get_tools_for_agent(agent_config: Dict[str, Any], available_tools: Dict[str, Any]) -> List[Any]:
    """
    Get the list of tool functions for an agent based on its configuration.

    Args:
        agent_config: Agent configuration
        available_tools: Dictionary of available tools

    Returns:
        List of tool functions for the agent
    """
    tools = []
    tool_names = agent_config.get('tools', [])

    for tool_name in tool_names:
        tool = get_tool_by_name(tool_name, available_tools)
        if tool:
            tools.append(tool)
        else:
            logger.warning(f"Tool '{tool_name}' not found in available tools")

    return tools


def save_results(results: Dict[str, Any], output_path: str) -> None:
    """
    Save results to a JSON file.

    Args:
        results: Results to save
        output_path: Path to save the results
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving results to {output_path}: {e}")


def init_results(save_dir, model_cfg_path, model_config, task_cfg_path, task_config):
    os.makedirs("/".join(os.path.join(save_dir, model_cfg_path).split("/")[:-1]), exist_ok=True)
    os.makedirs("/".join(os.path.join(save_dir, task_cfg_path).split("/")[:-1]), exist_ok=True)
    with open(os.path.join(save_dir, model_cfg_path), "w") as f:
        json.dump(model_config, f, indent=4)

    with open(os.path.join(save_dir, task_cfg_path), "w") as f:
        json.dump(task_config, f, indent=4)
