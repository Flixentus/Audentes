import torch
import torch.nn as nn
import json
import os 
import zipfile

script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, "config.json")

with open(config_path) as f:
    config = json.load(f)

DATA_DIR = os.path.join(script_dir, "data")
LOCAL_DIR = os.path.join(script_dir, "checkpoints")

torch.manual_seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    
os.makedirs(LOCAL_DIR, exist_ok=True)


def save_checkpoint_safely(checkpoint_data, filename):
    """Save locally first, verify it's a valid file, then copy to Drive.
    Never trust a save until it's been read back successfully."""
    local_path = os.path.join(LOCAL_DIR, filename)

    torch.save(checkpoint_data, local_path)

    try:
        with zipfile.ZipFile(local_path) as z:
            z.namelist()  # forces a real read, not just open
    except zipfile.BadZipFile:
        print(f"WARNING: {filename} failed integrity check after saving locally — NOT copying to Drive.")
        return False

    print(f"Checkpoint verified and saved: {filename}")
    return True


def load_checkpoint_safely(filename):
    """Try Drive first, fall back to local if Drive copy is bad."""
    if os.path.exists(filename):
        try:
            with zipfile.ZipFile(filename) as z:
                z.namelist()
            return torch.load(filename, map_location="cpu")
        except zipfile.BadZipFile:
            print(f"WARNING: {filename} is corrupted, trying next option...")
    return None


def get_gradient_norm(model):
    total_norm = 0.0

    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2

    return total_norm ** 0.5

def evaluate(model, loader, criterion, device):
    pass

    # 9arrebt wlh hh