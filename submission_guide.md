# Wunder Fund RNN Challenge — How to Make a Submission

This page covers the technical requirements for your submission, including the code format, how to package your files, and the resource limits.

***

## What to Submit

This is a code competition. You'll submit a `.zip` file containing all the code and artifacts needed to generate predictions.

**Key requirements:**
- The zip file must contain a `solution.py` file at its root.
- Your `solution.py` must define a class named `PredictionModel`.
- This class must have a `predict(self, data_point)` method.

### The `PredictionModel` Class

Required structure:

```python
import numpy as np
from utils import DataPoint

class PredictionModel:
    def __init__(self):
        # Initialize your model, load weights, etc.
        pass

    def predict(self, data_point: DataPoint) -> np.ndarray | None:
        # This is where your prediction logic goes.
        if not data_point.need_prediction:
            return None

        # When a prediction is needed, return a numpy array of length N.
        # Replace this with your model's actual output.
        prediction = np.zeros(data_point.state.shape)
        return prediction
```

- Your `predict` method receives a `DataPoint` object with the current market state.
- Your code should return `None` if `need_prediction` is `False`, and a NumPy array with your `N` feature predictions otherwise.

**DataPoint Object Attributes:**
- `seq_ix: int`: ID for the current sequence.
- `step_in_seq: int`: Step number within the sequence.
- `need_prediction: bool`: Whether a prediction is required for this point.
- `state: np.ndarray`: Current market state vector of N features.

> **NOTE:**  
> Remember to handle the model's internal state. When you encounter a new sequence (`seq_ix`), you must reset any recurrent state.

***

### Including Other Files

You can include additional files in your zip archive such as:
- Model weights (`.pt`, `.h5`, `.onnx`, etc.)
- Helper Python modules (`.py` files)
- Configuration files (`.json`, `.yaml`)
- Small data files

**Ensure:** `solution.py` is at the root of the archive.

***

## How to Package Your Solution

Package all files into a single `.zip` archive.

**macOS/Linux/Windows:**
1. Open a terminal.
2. `cd` into your solution directory.
3. Run:

```bash
zip -r ../submission.zip .
```

This creates `submission.zip` in the parent directory, containing everything in the folder.

***

## How Submissions Are Scored

### Docker

- When you submit, a scoring Docker container is deployed.
- Base image: `python:3.11-slim-bookworm`
- ML libraries are configured to prevent network access.

**Relevant Dockerfile section:**

```dockerfile
FROM python:3.11-slim-bookworm
RUN apt-get update && apt-get full-upgrade -y
RUN apt-get install -y curl libgomp1 p7zip-full build-essential && apt-get clean && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip \
    && python -m pip install --prefer-binary --extra-index-url https://download.pytorch.org/whl/cpu -r /tmp/requirements.txt \
    && python -m pip install orbax-checkpoint \
    && python -m pip check && pip cache purge
# Keep heavy libs strictly offline at runtime
ENV HF_HUB_DISABLE_TELEMETRY=1 \
    TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
    HF_HUB_OFFLINE=1 \
    WANDB_DISABLED=1 MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=false \
    HOME=/app \
    XDG_CACHE_HOME=/app/.cache \
    MPLCONFIGDIR=/app/.matplotlib \
    TORCH_HOME=/app/.cache/torch \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface/transformers \
    HF_DATASETS_CACHE=/app/.cache/huggingface/datasets \
    NUMBA_CACHE_DIR=/app/.cache/numba
### R̴̨̋E̴̟͝S̸̪̚T̸̢͘ ̶̜̈́I̴͖͗S̷̢͗ ̴̘̂C̶̕͜E̶̋͜N̵̼̓S̴̙͠O̴̘͐R̶̼͑Ě̵͕Ḓ̵̋ ###
```

***

### Libs and Packages

- `requirements.txt` is referenced in the Dockerfile.
- The environment includes a reasonable set of popular ML libraries.
- If you need additional packages, contact via Discord or [get help](https://wundernn.io/docs/get_help).

***

## Resource and Time Limits

Your code will run in an isolated environment under these constraints:
- **No internet access**
- **Time limit:** 60 minutes for generating predictions for the entire test set
- **No GPU**
- **CPU:** 1 core

***
---

[1](https://wundernn.io/docs/how_to_submit)
