import torch
import torch.nn as nn
import yaml
from pathlib import Path
import zipfile

from Audentes import Audentes # Jme3t Blocks kamlin f class smitha Audentes, katpassi liha config file bo7do 👍

script_dir = Path(__file__).parent
config_path = script_dir / "config.yaml"

with open(config_path) as f:
    config = yaml.safe_load(f)

DATA_DIR = script_dir / "data"
LOCAL_DIR = script_dir / "checkpoints"

torch.manual_seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    
LOCAL_DIR.mkdir(parents=True, exist_ok=True)


def save_checkpoint_safely(checkpoint_data, filename):
    """Save locally first, verify it's a valid file, then copy to Drive.
    Never trust a save until it's been read back successfully."""
    local_path = LOCAL_DIR / filename

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
    if (LOCAL_DIR / filename).is_file():
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