# Tourism Project - MLOps Pipeline

This project implements an end-to-end MLOps pipeline for predicting tourism package purchases.

## Project Structure
```
tourism_project/
├── .github/workflows/pipeline.yml  # GitHub Actions Workflow
├── data/                           # Local data storage
├── src/
│   ├── data_registration.py        # Upload raw data to HF
│   ├── data_preparation.py         # Process and split data
│   └── model_building.py           # Train and register model
├── deployment/
│   ├── app.py                      # Streamlit application
│   ├── Dockerfile                  # Docker configuration
│   ├── requirements.txt            # App dependencies
│   └── deploy.py                   # HF Space deployment script
└── requirements.txt                # Project dependencies
```

## Setup Instructions

1.  **Hugging Face Token**:
    -   Get your Write Access Token from [Hugging Face Settings](https://huggingface.co/settings/tokens).
    -   Set it as an environment variable `HF_TOKEN` locally.
    -   Add it to your GitHub Repository Secrets as `HF_TOKEN`.
    -   Add your HF Username as `HF_USERNAME` in Secrets as well.

2.  **Run Locally**:
    ```bash
    # Install dependencies
    pip install -r requirements.txt
    
    # 1. Register Data
    python src/data_registration.py
    
    # 2. Prepare Data
    python src/data_preparation.py
    
    # 3. Build Model
    python src/model_building.py
    
    # 4. Run App
    streamlit run deployment/app.py
    ```

3.  **Deployment**:
    -   The GitHub Actions pipeline will automatically run on push to `main`.
    -   It will orchestrate the entire flow and deploy the app to your Hugging Face Space.
    -   You can also run `python deployment/deploy.py` manually.

## Rubric Fulfillment
-   **Data Registration**: `src/data_registration.py` handles upload.
-   **Data Preparation**: `src/data_preparation.py` handles cleaning, splitting, and re-upload.
-   **Model Building**: `src/model_building.py` trains RF, evaluations, and registers to Model Hub.
-   **Deployment**: Dockerized Streamlit app in `deployment/` folder.
-   **MLOps**: `.github/workflows/pipeline.yml` automates the process.
