import os
import yaml

LAB_PATH = os.path.abspath(os.path.dirname(__file__))
CONFIG_FILE = os.path.join(os.path.dirname(__file__), 'config.yaml')

def get_config():
    try:
        with open(CONFIG_FILE, 'r') as file:
            configS = yaml.safe_load(file)
            return configS
    except FileNotFoundError:
        print(f"Config file not found at: {CONFIG_FILE}")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error reading config: {e}")
        return None


config = get_config()
