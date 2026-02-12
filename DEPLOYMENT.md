# Deployment Guide for Tourism Project

**Project**: `RavikanthAI9/tourism_project`
**Space URL**: [https://huggingface.co/spaces/RavikanthAI9/tourism_project](https://huggingface.co/spaces/RavikanthAI9/tourism_project)

## ⚠️ Critical Configuration Step
Before deploying, you **MUST** configure the project:

1.  **Check `config.yaml`**: Ensure `hf_username` and file paths are correct.
2.  **Create `.env` file**: (Local deployment only)
    Create a file named `.env` in the root folder with:
    ```
    HF_TOKEN=your_write_token_here
    HF_USERNAME=RavikanthAI9
    ```

---

## Option 1: Automatic Deployment (GitHub Actions)
**Recommended for MLOps** - Runs automatically on push.

1.  **Configure Secrets on GitHub**:
    *   Go to **Settings** -> **Secrets and variables** -> **Actions**.
    *   Add `HF_TOKEN`: Your Hugging Face Write Token.
    *   Add `HF_USERNAME`: `RavikanthAI9`.

2.  **Push to GitHub**:
    ```bash
    git add .
    git commit -m "Update configuration"
    git push origin main
    ```
    The pipeline defined in `.github/workflows/pipeline.yml` will run automatically.

---

## Option 2: Local Deployment
**Run from your machine.**

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run Deployment Script**:
    ```bash
    python deployment/deploy.py
    ```
    *   This reads settings from `config.yaml`.
    *   It creates/updates the Space `RavikanthAI9/tourism_project`.
    *   It uploads the `deployment/` folder.

---

## Option 3: Manual Drag & Drop
1.  Create a Space on Hugging Face (SDK: Docker).
2.  Upload `Dockerfile`, `app.py`, `requirements.txt`, and `config.yaml` to the Files tab.
3.  **Important**: Add `HF_TOKEN` and `HF_USERNAME` to the Space's **Settings > Secrets**.

## Troubleshooting
*   **Missing Model**: Run `python src/model_building.py` locally first if testing locally to ensure `model.pkl` exists in your HF Model Hub.
*   **Port Match**: Dockerfile uses port `7860`.
