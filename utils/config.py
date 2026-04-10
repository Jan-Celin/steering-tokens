import yaml

def load_config(path_to_yaml: str) -> dict:
    with open(path_to_yaml, 'r') as file:
        return yaml.safe_load(file)
