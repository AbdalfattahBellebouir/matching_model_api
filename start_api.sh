#!/bin/bash
# conda activate automatching
python -m uvicorn matching_model_api:app --host 0.0.0.0 --port 7777
