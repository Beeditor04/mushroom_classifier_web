# through .env keys
import wandb
import os
from dotenv import load_dotenv
load_dotenv()
import argparse

def create_parser():
    parser = argparse.ArgumentParser(description="Train with different models.")
    parser.add_argument("--online", default=1, type=int)
    return parser

def parse_args():
    # sys.argv = [sys.argv[0]]
    parser = create_parser()
    args, unknown = parser.parse_known_args()
    return args

args = parse_args()
api_key = None
if args.online:
    print("Running online...")
    os.environ["WANDB_BASE_URL"] = "https://api.wandb.ai"
    api_key = os.getenv("WANDB_API_KEY")
    print(api_key)
else:
    print("Runnning offline...")
    os.environ["WANDB_BASE_URL"] = "http://localhost:8080"
    api_key = os.getenv("LOCAL_WANDB")
print(api_key)
wandb.login(key=api_key, relogin=True)
wandb.require("core")