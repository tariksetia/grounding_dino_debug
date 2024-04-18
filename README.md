# Grounding Dino Debug

### Running API

```bash

python -m virtualenv .venv
source .venv/bin/activate

pip install -q git+https://github.com/IDEA-Research/GroundingDINO.git
pip install -q git+https://github.com/huggingface/transformers.git
pip install -q opencv-python pillow
pip install -q fastapi[all]
pip install -q ipykernel

uvicorn main:app --reload
```

Call Endpoint: `curl http://0.0.0.0:8000/video`

### Running Notebook

open the notebook in VSCode and select the virtual env as kernel
