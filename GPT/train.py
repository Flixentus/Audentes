import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import os
import pickle
from audentes import Audentes
import zipfile
from torch.amp import autocast, GradScaler
import json
import time
from tokenizers import Tokenizer as HFTokenizer
from bpe_tokenizer import BPETokenizerWrapper
import yaml


script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, "config.yaml")


with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)["Model"]

DATA_DIR = os.path.join(script_dir, "data")
LOCAL_DIR = os.path.join(script_dir, "checkpoints")

tokenizer_path = os.path.join(
    DATA_DIR,
    "chat_bpe_tokenizer.json"
)

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


class FinancialTextDataset(Dataset):
    """Dataset wrapper for the merged financial datasets."""
    
    def __init__(self, data_list, tokenizer):
        """
        Args:
            data_list: List of dicts with 'text' and 'source' keys
            tokenizer: HuggingFace tokenizer for encoding
        """
        self.data = data_list
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]
        
        # Tokenize the text
        encoded = self.tokenizer.encode(text)
        tokens = torch.tensor(encoded.ids, dtype=torch.long)
        
        return tokens


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_tokens = 0

    with torch.no_grad():

        for tgt in loader:
            tgt = tgt.to(device)
            decoder_input = tgt[:, :-1]
            target = tgt[:, 1:]

            logits = model(tgt, decoder_input)  

            predictions = logits.argmax(dim=-1)
            mask = target != 0
            correct = ((predictions ==  target) & mask).sum().item()
            total = mask.sum().item()

            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                target.reshape(-1)
                )
            total_loss += loss.item()
            total_correct += correct
            total_tokens += total

        avg_loss = total_loss / len(loader)
        accuracy = total_correct / total_tokens if total_tokens > 0 else 0
        model.train()
        return avg_loss, accuracy

def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    train_pkl_path = os.path.join(DATA_DIR, "ultrachat_train.pkl")
    test_pkl_path = os.path.join(DATA_DIR, "ultrachat_test.pkl")

    with open(train_pkl_path, "rb") as f:
        train_data = pickle.load(f)

    with open(test_pkl_path, "rb") as f:
        test_data = pickle.load(f)
    
    hf_tokenizer = HFTokenizer.from_file(tokenizer_path)
    actual_vocab_size = hf_tokenizer.get_vocab_size()

    config["vocab_size"] = actual_vocab_size

    print(f"Loaded tokenizer: {tokenizer_path}")
    print(f"Actual vocab size: {actual_vocab_size}")

    train_dataset = FinancialTextDataset(train_data, hf_tokenizer)
    test_dataset = FinancialTextDataset(test_data, hf_tokenizer)

    loader = DataLoader(
        train_dataset,
        batch_size=config["BATCH_SIZE"],
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["BATCH_SIZE"],
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    model = Audentes(
        actual_vocab_size,
        config["d_model"],
        config["num_heads"],
        config["d_ff"],
        0,
        config["num_decoder_layers"],
    ).to(device)

    total_params = sum(
        p.numel() for p in model.parameters()
    )

    experiment_config = {
        "model_type": config["model_type"],
        "d_model": config["d_model"],
        "layers": {
            "encoder": config["num_encoder_layers"],
            "decoder": config["num_decoder_layers"]
        },
        "vocab_size": actual_vocab_size,
        "parameters": total_params
    }

    json.dump(
        experiment_config,
        open(os.path.join(LOCAL_DIR,"experiment_config.json"),"w"),
        indent=2
    ) # hadi gha bach nb9aw msavin lconfig dial current experiment

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["eps"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
    scaler = GradScaler("cuda", enabled=torch.cuda.is_available())
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    start_epoch = 0
    start_batch_idx = 0
    checkpoint = load_checkpoint_safely(os.path.join(LOCAL_DIR, "checkpoint_latest.pt"))

    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        start_batch_idx = 0

        print(f"Resuming from epoch {start_epoch + 1}, batch {start_batch_idx}")
    else:
        print("No valid checkpoint found — starting fresh from epoch 1.")

    for epoch in range(start_epoch, config["epochs"]):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        epoch_start_time = time.time()
        model.train()
        running_loss = 0
        total_correct = 0
        total_tokens = 0

        tokens_processed = 0
        gradient_norm_sum = 0
        gradient_steps = 0

        optimizer.zero_grad(set_to_none=True)
        last_grad_norm = 0.0
        for batch_idx, (x, y) in enumerate(loader):
            if epoch == start_epoch and batch_idx < start_batch_idx:
                continue

            x= x.to(device)
            y= y.to(device)

            tokens_processed += (x != 0).sum().item()
            tokens_processed += (y != 0).sum().item()


            with autocast(
                "cuda",
                dtype=torch.bfloat16,
                enabled=torch.cuda.is_available()
            ):
                logits = model(x)
            
                loss = criterion(
                    logits.reshape(-1, logits.size(-1)),
                    y.reshape(-1)
                )
            
            
            loss = loss / config["accumulation_steps"]
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % config["accumulation_steps"] == 0:
            
                scaler.unscale_(optimizer)
            
                grad_norm = get_gradient_norm(model)
                last_grad_norm = grad_norm
                gradient_norm_sum += grad_norm
                gradient_steps += 1
            
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0
                )
            
                scaler.step(optimizer)
                scaler.update()
            
                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item()

            predictions = logits.argmax(dim=-1)
            mask = y != 0
            correct = ((predictions == y) & mask).sum().item()
            total = mask.sum().item()

            total_correct += correct
            total_tokens += total
            accuracy = total_correct / total_tokens if total_tokens > 0 else 0

            print(
                f"Batch {batch_idx + 1} | "
                f"Epoch {epoch+1} | "
                f"Loss: {running_loss * 2 / (batch_idx + 1):.4f} | "
                f"Accuracy: {accuracy:.2%} | "
                f"Grad Norm: {last_grad_norm:.4f}"
            )

        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        perplexity = torch.exp(torch.tensor(test_loss))
        print(f"[Eval] Epoch {epoch+1} | test Loss: {test_loss:.4f} | test Accuracy: {test_acc:.2%} | test PPL: {perplexity:.2f}")
        scheduler.step(test_loss)
        print(optimizer.param_groups[0]['lr'])

        epoch_elapsed = time.time() - epoch_start_time  
        tokens_per_second = tokens_processed / epoch_elapsed

        avg_gradient_norm = (
            gradient_norm_sum / gradient_steps
            if gradient_steps > 0
            else 0
        )

        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)
        else:
            peak_memory = 0      

        hours, rem = divmod(epoch_elapsed, 3600)
        minutes, seconds = divmod(rem, 60)
        epoch_time_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"  
        train_loss_last = running_loss / len(loader)
        train_acc_last = total_correct / total_tokens if total_tokens > 0 else 0

        results_line = (
            f"Epoch {epoch+1} | "
            f"Total parameters: {total_params/1e6:.2f}M | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Accuracy: {test_acc:.2%} | "
            f"Perplexity: {perplexity:.2f} | "
            f"Train Loss: {train_loss_last:.4f} | "
            f"Train Accuracy: {train_acc_last:.2%} | "
            f"Time: {epoch_time_str} | "
            f"Tokens/sec: {tokens_per_second:.2f} | "
            f"Grad Norm: {avg_gradient_norm:.4f}\n"
        )

        local_results = os.path.join(LOCAL_DIR, "epoch_results.txt")
        with open(local_results, "a") as f:
            f.write(results_line)

        checkpoint_data = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch
        }
        save_checkpoint_safely(checkpoint_data, "checkpoint_latest.pt")

    torch.save(
        model.state_dict(),
        os.path.join(LOCAL_DIR,"final_model.pt")
    )