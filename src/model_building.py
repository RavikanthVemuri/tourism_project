import os
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from huggingface_hub import hf_hub_download, HfApi, create_repo
import joblib
import yaml
from dotenv import load_dotenv

# Load configuration
load_dotenv()
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Constants
HF_USERNAME = os.getenv("HF_USERNAME", config['project']['hf_username'])
PROCESSED_DATASET_REPO = f"{HF_USERNAME}/{config['data']['processed_dataset_name']}"
MODEL_REPO = f"{HF_USERNAME}/{config['model']['model_name']}"
MODEL_FILENAME = config['model']['filename']
COLUMNS_FILENAME = config['model']['columns_filename']

def build_model():
    print("Starting model building...")
    hf_token = os.getenv("HF_TOKEN")
    
    try:
        # Load Data
        print("Downloading data...")
        try:
            train_path = hf_hub_download(repo_id=PROCESSED_DATASET_REPO, filename="train.csv", repo_type="dataset", token=hf_token)
            test_path = hf_hub_download(repo_id=PROCESSED_DATASET_REPO, filename="test.csv", repo_type="dataset", token=hf_token)
        except Exception as e:
            print(f"Failed to download data: {e}")
            return

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        # Prepare X and y
        target_col = config['data']['target_column']
        X_train = train_df.drop(target_col, axis=1)
        y_train = train_df[target_col]
        X_test = test_df.drop(target_col, axis=1)
        y_test = test_df[target_col]
        
        # Save columns for deployment
        columns = X_train.columns.tolist()
        joblib.dump(columns, COLUMNS_FILENAME)
        print(f"Feature columns saved to {COLUMNS_FILENAME}")
        
        # Train Model
        print("Training Random Forest...")
        rf_params = config['model']['params']
        rf = RandomForestClassifier(**rf_params)
        rf.fit(X_train, y_train)
        
        # Evaluate
        print("Evaluating...")
        y_pred = rf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        print(f"Accuracy: {accuracy}")
        print(report)
        
        with open("metrics.txt", "w") as f:
            f.write(f"Accuracy: {accuracy}\n\n")
            f.write(report)
            
        # Save Model
        joblib.dump(rf, MODEL_FILENAME)
        
        # Upload to HF
        print("Uploading artifacts to Hugging Face...")
        api = HfApi()
        create_repo(repo_id=MODEL_REPO, repo_type="model", exist_ok=True, token=hf_token)
        
        for file in [MODEL_FILENAME, COLUMNS_FILENAME, "metrics.txt"]:
            api.upload_file(
                path_or_fileobj=file,
                path_in_repo=file,
                repo_id=MODEL_REPO,
                repo_type="model",
                token=hf_token
            )
        print("Model and artifacts uploaded successfully.")
        
    except Exception as e:
        print(f"Error in model building: {e}")

if __name__ == "__main__":
    build_model()
