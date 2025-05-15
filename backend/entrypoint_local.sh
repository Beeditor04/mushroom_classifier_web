#!/bin/bash
python login_wandb.py
uvicorn app:app --host 0.0.0.0 --port 8000