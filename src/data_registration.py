import os
import pandas as pd
from huggingface_hub import HfApi, create_repo
import yaml
from dotenv import load_dotenv

# Load configuration
load_dotenv()
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Constants from config
HF_USERNAME = os.getenv("HF_USERNAME", config['project']['hf_username'])
DATASET_NAME = config['data']['dataset_name']
LOCAL_DATA_PATH = config['data']['raw_file']
REPO_ID = f"{HF_USERNAME}/{DATASET_NAME}"

def register_data():
    """
    Registers the local dataset to Hugging Face.
    """
    print(f"Registering data to {REPO_ID}...")
    
    # Check if file exists
    if not os.path.exists(LOCAL_DATA_PATH):
        raise FileNotFoundError(f"File not found at {LOCAL_DATA_PATH}. Please populate the data folder.")
    
    # Authentication check
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("Warning: HF_TOKEN environment variable not set. Operations requiring authentication might fail.")

    api = HfApi()
    
    try:
        # Create repo if it doesn't exist
        create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True, token=hf_token)
        print(f"Repository {REPO_ID} created or already exists.")
        
        # Upload file
        api.upload_file(
            path_or_fileobj=LOCAL_DATA_PATH,
            path_in_repo="tourism (1).csv",
            repo_id=REPO_ID,
            repo_type="dataset",
            token=hf_token
        )
        print(f"Successfully uploaded {LOCAL_DATA_PATH} to {REPO_ID}.")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    register_data()
