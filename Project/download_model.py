#!/usr/bin/env python3
"""Download trained RL policy from wandb run."""

import wandb
import os

WANDB_PROJECT = "damiane/pupperv3-mjx-rl"
RUN_ID = "g7t28tdo"

def download_model():
    """Download model files and artifacts from wandb run."""
    # Interactive login to wandb
    wandb.login(relogin=True)
    
    # Get the run
    api = wandb.Api()
    run = api.run(f"{WANDB_PROJECT}/{RUN_ID}")
    
    # Create models directory if it doesn't exist
    model_dir = os.path.join(os.path.dirname(__file__), "models", RUN_ID)
    os.makedirs(model_dir, exist_ok=True)
    
    # Download all files from the run
    print(f"Downloading files from run {RUN_ID}...")
    for file in run.files():
        # Skip large video files
        if 'video' in file.name or 'wandb_manifest' in file.name:
            print(f"  Skipping {file.name}")
            continue
        print(f"  Downloading {file.name}...")
        try:
            file.download(root=model_dir, replace=True)
        except Exception as e:
            print(f"  Warning: {e}")
    
    # Download artifacts from the run (actual model weights)
    print(f"\nDownloading artifacts...")
    for artifact_name in run.logged_artifacts():
        print(f"  Found artifact: {artifact_name.name} (type: {artifact_name.type})")
        artifact_dir = os.path.join(model_dir, "artifacts", artifact_name.name)
        os.makedirs(artifact_dir, exist_ok=True)
        
        try:
            artifact = api.artifact(f"{WANDB_PROJECT}/{artifact_name.name}")
            artifact.download(artifact_dir)
            print(f"    ✓ Downloaded to {artifact_dir}")
        except Exception as e:
            print(f"    Warning: {e}")
    
    print(f"\n✓ Complete! Model files are in {model_dir}")

if __name__ == "__main__":
    download_model()
