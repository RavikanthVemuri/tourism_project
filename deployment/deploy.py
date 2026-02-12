import os
from huggingface_hub import HfApi
import yaml
from dotenv import load_dotenv

# Load configuration
load_dotenv()
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Constants
HF_USERNAME = os.getenv("HF_USERNAME", config['project']['hf_username'])
SPACE_ID = f"{HF_USERNAME}/{config['deployment']['app_name']}"

def deploy_to_space():
    print(f"Deploying to Hugging Face Space: {SPACE_ID}...")
    hf_token = os.getenv("HF_TOKEN")
    
    if not hf_token:
        print("Error: HF_TOKEN not found.")
        return

    api = HfApi()
    
    try:
        # Create Space if not exists (sdk="docker")
        api.create_repo(
            repo_id=SPACE_ID,
            repo_type="space",
            space_sdk="docker",
            exist_ok=True,
            token=hf_token
        )
        print(f"Space {SPACE_ID} is ready.")
        
        # Upload folder
        print("Uploading deployment files...")
        api.upload_folder(
            folder_path="deployment",
            repo_id=SPACE_ID,
            repo_type="space",
            token=hf_token
        )
        
        # Upload config.yaml explicitly
        print("Uploading config.yaml...")
        api.upload_file(
            path_or_fileobj="config.yaml",
            path_in_repo="config.yaml",
            repo_id=SPACE_ID,
            repo_type="space",
            token=hf_token
        )
        print("Deployment files uploaded. The Space should receive a rebuild trigger.")
        
    except Exception as e:
        print(f"Deployment failed: {e}")

if __name__ == "__main__":
    deploy_to_space()
