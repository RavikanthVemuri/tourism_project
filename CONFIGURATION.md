# Configuration Guide

The project now uses dynamic configuration.

## `config.yaml`
This file contains all project parameters. You can change model parameters, dataset names, or deployment settings here without touching the code.

```yaml
project:
  name: "tourism_project"
  hf_username: "RavikanthAI9"  # Change this if your username changes

data:
  raw_file: "data/tourism.csv"
  ...
model:
  type: "RandomForest"
  params:
    n_estimators: 100
    ...
```

## `.env` File
This file stores your secrets.  **DO NOT COMMIT THIS TO GITHUB.**
Create a file named `.env` in the `tourism_project` root with:

```
HF_TOKEN=your_huggingface_write_token_here
HF_USERNAME=RavikanthAI9
```

The scripts will automatically load these variables.
