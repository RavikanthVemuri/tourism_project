import yaml
import os
from dotenv import load_dotenv

def load_config(config_path="config.yaml"):
    """Loads the YAML configuration file."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def load_env():
    """Loads environment variables from .env file."""
    load_dotenv()
