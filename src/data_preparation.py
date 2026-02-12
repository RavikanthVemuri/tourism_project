import os
import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import hf_hub_download, HfApi, create_repo
import numpy as np
import yaml
from dotenv import load_dotenv

# Load configuration
load_dotenv()
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Constants from config
HF_USERNAME = os.getenv("HF_USERNAME", config['project']['hf_username'])
RAW_DATASET_REPO = f"{HF_USERNAME}/{config['data']['dataset_name']}"
PROCESSED_DATASET_REPO = f"{HF_USERNAME}/{config['data']['processed_dataset_name']}"
RAW_FILENAME = "tourism (1).csv"
TEST_SIZE = config['data']['test_size']
RANDOM_STATE = config['data']['random_state']

def prepare_data():
    print("Starting data preparation...")
    
    hf_token = os.getenv("HF_TOKEN")
    
    try:
        # 1. Load dataset from Hugging Face
        print(f"Downloading {RAW_FILENAME} from {RAW_DATASET_REPO}...")
        local_path = hf_hub_download(
            repo_id=RAW_DATASET_REPO,
            filename=RAW_FILENAME,
            repo_type="dataset",
            token=hf_token
        )
        df = pd.read_csv(local_path)
        print("Dataset loaded successfully.")
        
        # 2. Data Cleaning
        print("Cleaning data...")
        # Drop unnecessary ID column
        if 'CustomerID' in df.columns:
            df = df.drop(columns=['CustomerID'])
            
        # Handle missing values
        # Impute numerical columns with median
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            df[col] = df[col].fillna(df[col].median())
            
        # Impute categorical columns with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0])
            
        # 3. Feature Engineering / Encoding (basic for now)
        # Convert categorical variables to dummy variables
        df = pd.get_dummies(df, drop_first=True)
        
        # 4. Split Data
        print("Splitting data...")
        target_col = config['data']['target_column']
        X = df.drop(target_col, axis=1) # Assuming 'ProdTaken' is the target
        y = df[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        
        # Combine X and y for saving (easier for upload)
        train_df = pd.concat([X_train, y_train], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)
        
        # Save locally
        os.makedirs("data/processed", exist_ok=True)
        train_path = "data/processed/train.csv"
        test_path = "data/processed/test.csv"
        
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        print("Train and Test sets saved locally.")
        
        # 5. Upload to Hugging Face
        print(f"Uploading processed data to {PROCESSED_DATASET_REPO}...")
        api = HfApi()
        create_repo(repo_id=PROCESSED_DATASET_REPO, repo_type="dataset", exist_ok=True, token=hf_token)
        
        api.upload_file(
            path_or_fileobj=train_path,
            path_in_repo="train.csv",
            repo_id=PROCESSED_DATASET_REPO,
            repo_type="dataset",
            token=hf_token
        )
        api.upload_file(
            path_or_fileobj=test_path,
            path_in_repo="test.csv",
            repo_id=PROCESSED_DATASET_REPO,
            repo_type="dataset",
            token=hf_token
        )
        print("Processed data uploaded successfully.")
        
    except Exception as e:
        print(f"An error occurred during data preparation: {e}")

if __name__ == "__main__":
    prepare_data()
