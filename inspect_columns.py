import joblib
from huggingface_hub import hf_hub_download
import yaml
import os
from dotenv import load_dotenv

load_dotenv()
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

HF_USERNAME = os.getenv("HF_USERNAME", config['project']['hf_username'])
MODEL_REPO = f"{HF_USERNAME}/{config['model']['model_name']}"
COLUMNS_FILENAME = config['model']['columns_filename']

print(f"Downloading {COLUMNS_FILENAME} from {MODEL_REPO}...")
columns_path = hf_hub_download(repo_id=MODEL_REPO, filename=COLUMNS_FILENAME)
columns = joblib.load(columns_path)

print("\n--- MODEL COLUMNS ---")
with open("columns.txt", "w") as f:
    for col in columns:
        f.write(f"{col}\n")
print("Columns written to columns.txt")
