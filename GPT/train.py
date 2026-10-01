import os
import time
import pickle
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler
from tokenizers import Tokenizer

import yaml

from Audentes import Audentes


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG_PATH = os.path.join(
    SCRIPT_DIR,
    "..",
    "config.yaml"
)

DATA_DIR = os.path.join(
    SCRIPT_DIR,
    "data"
)

CHECKPOINT_DIR = "/content/drive/MyDrive/Audentes/checkpoints"

TOKENIZER_PATH = os.path.join(
    DATA_DIR,
    "chat_bpe_tokenizer.json"
)

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)["Model"]


torch.manual_seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


class ChatDataset(Dataset):

    def __init__(self, data, tokenizer, seq_len=512):
        self.data = data
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        tokens = self.tokenizer.encode(
            self.data[idx]["text"]
        ).ids

        tokens = torch.tensor(
            tokens,
            dtype=torch.long
        )

        if len(tokens) < self.seq_len:

            tokens = F.pad(
                tokens,
                (0, self.seq_len - len(tokens)),
                value=0
            )

        else:
            tokens = tokens[:self.seq_len]

        return tokens[:-1], tokens[1:]


def get_gradient_norm(model):

    total = 0.0

    for parameter in model.parameters():

        if parameter.grad is not None:

            total += (
                parameter.grad.detach()
                .norm(2)
                .item()
                ** 2
            )

    return total ** 0.5


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch
):

    path = os.path.join(
        CHECKPOINT_DIR,
        "checkpoint_latest.pt"
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch
        },
        path
    )

    print(f"Checkpoint saved to: {path}")


def evaluate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    with torch.no_grad():

        for x, y in loader:

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=device.type == "cuda"
            ):

                logits, _, _, _, _ = model(x)

                loss = criterion(
                    logits.reshape(-1, logits.size(-1)),
                    y.reshape(-1)
                )

            predictions = logits.argmax(dim=-1)

            mask = y != 0

            total_correct += (
                (predictions == y) & mask
            ).sum().item()

            total_tokens += mask.sum().item()

            total_loss += loss.item()

    avg_loss = total_loss / len(loader)

    accuracy = (
        total_correct / total_tokens
        if total_tokens
        else 0.0
    )

    model.train()

    return avg_loss, accuracy

def train():

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print("CUDA:", torch.cuda.is_available())

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

        print(
            "VRAM:",
            round(
                torch.cuda.get_device_properties(0)
                .total_memory / 1024**3,
                2
            ),
            "GB"
        )

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


    with open(
        os.path.join(DATA_DIR, "ultrachat_train.pkl"),
        "rb"
    ) as f:
        train_data = pickle.load(f)


    with open(
        os.path.join(DATA_DIR, "ultrachat_test.pkl"),
        "rb"
    ) as f:
        test_data = pickle.load(f)


    tokenizer = Tokenizer.from_file(
        TOKENIZER_PATH
    )

    vocab_size = tokenizer.get_vocab_size()

    print("Train examples:", len(train_data))
    print("Test examples:", len(test_data))
    print("Vocabulary:", vocab_size)


    train_dataset = ChatDataset(
        train_data,
        tokenizer,
        seq_len=512
    )

    test_dataset = ChatDataset(
        test_data,
        tokenizer,
        seq_len=512
    )


    batch_size = int(
        config["batch_size"]
    )


    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )


    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )


    model = Audentes(
        CONFIG_PATH,
        vocab_size
    ).to(device)


    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Parameters: {total_params / 1e6:.2f}M"
    )


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=0.1
    )


    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=1
    )


    scaler = GradScaler(
        "cuda",
        enabled=device.type == "cuda"
    )


    criterion = nn.CrossEntropyLoss(
        ignore_index=0
    )


    start_epoch = 0

    checkpoint_path = os.path.join(
        CHECKPOINT_DIR,
        "checkpoint_latest.pt"
    )


    if os.path.exists(checkpoint_path):

        print("Loading checkpoint...")

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu"
        )


        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )


        start_epoch = checkpoint["epoch"] + 1


        print(
            f"Resuming from epoch {start_epoch}"
        )

    else:

        print(
            "Starting fresh training."
        )


    accumulation_steps = int(
        config["accumulation_steps"]
    )


    for epoch in range(
        start_epoch,
        int(config["epochs"])
    ):

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()


        model.train()

        epoch_start = time.time()

        running_loss = 0.0

        total_correct = 0
        total_tokens = 0

        gradient_norm_sum = 0.0
        gradient_steps = 0

        optimizer.zero_grad(
            set_to_none=True
        )
        for batch_idx, (x, y) in enumerate(train_loader):

            x = x.to(
                device,
                non_blocking=True
            )

            y = y.to(
                device,
                non_blocking=True
            )


            with autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=device.type == "cuda"
            ):

                (
                    logits,
                    _,
                    aux_loss,
                    z_loss,
                    drop_rate
                ) = model(x)


                loss = criterion(
                    logits.reshape(
                        -1,
                        logits.size(-1)
                    ),
                    y.reshape(-1)
                )


            scaled_loss = (
                loss / accumulation_steps
            )

            scaler.scale(
                scaled_loss
            ).backward()


            if (
                (batch_idx + 1) % accumulation_steps == 0
                or batch_idx + 1 == len(train_loader)
            ):

                scaler.unscale_(
                    optimizer
                )


                grad_norm = get_gradient_norm(
                    model
                )


                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0
                )


                scaler.step(
                    optimizer
                )

                scaler.update()


                optimizer.zero_grad(
                    set_to_none=True
                )


                gradient_norm_sum += grad_norm
                gradient_steps += 1


            running_loss += loss.item()


            predictions = logits.argmax(
                dim=-1
            )

            mask = y != 0


            total_correct += (
                (predictions == y) & mask
            ).sum().item()


            total_tokens += mask.sum().item()


            accuracy = (
                total_correct / total_tokens
                if total_tokens
                else 0.0
            )


            print(
                f"Epoch {epoch + 1} | "
                f"Batch {batch_idx + 1}/{len(train_loader)} | "
                f"Loss {loss.item():.4f} | "
                f"Accuracy {accuracy:.2%} | "
                f"Drop {drop_rate:.2%}"
            )


        train_loss = (
            running_loss / len(train_loader)
        )


        train_accuracy = (
            total_correct / total_tokens
            if total_tokens
            else 0.0
        )


        test_loss, test_accuracy = evaluate(
            model,
            test_loader,
            criterion,
            device
        )


        scheduler.step(
            test_loss
        )


        history_path = os.path.join(
            CHECKPOINT_DIR,
            "training_history.json"
        )

        history = []

        if os.path.exists(history_path):

            with open(
                history_path,
                "r"
            ) as f:
                history = json.load(f)


        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "test_loss": test_loss,
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy
            }
        )


        with open(
            history_path,
            "w"
        ) as f:
            json.dump(
                history,
                f,
                indent=2
            )


        perplexity = torch.exp(
            torch.tensor(test_loss)
        ).item()


        elapsed = (
            time.time() - epoch_start
        )


        avg_grad = (
            gradient_norm_sum / gradient_steps
            if gradient_steps
            else 0.0
        )


        peak_memory = (
            torch.cuda.max_memory_allocated()
            / 1024**3
            if device.type == "cuda"
            else 0.0
        )


        print("\n" + "=" * 60)

        print(
            f"Epoch {epoch + 1} complete"
        )

        print(
            f"Train Loss: {train_loss:.4f}"
        )

        print(
            f"Train Accuracy: {train_accuracy:.2%}"
        )

        print(
            f"Test Loss: {test_loss:.4f}"
        )

        print(
            f"Test Accuracy: {test_accuracy:.2%}"
        )

        print(
            f"Perplexity: {perplexity:.2f}"
        )

        print(
            f"Learning Rate: {optimizer.param_groups[0]['lr']}"
        )

        print(
            f"Average Grad Norm: {avg_grad:.4f}"
        )

        print(
            f"Peak VRAM: {peak_memory:.2f} GB"
        )

        print(
            f"Time: {elapsed:.1f}s"
        )

        print("=" * 60 + "\n")


        save_checkpoint(
            model,
            optimizer,
            scheduler,
            epoch
        )


    final_path = os.path.join(
        CHECKPOINT_DIR,
        "final_model.pt"
    )


    torch.save(
        model.state_dict(),
        final_path
    )


    print("Training complete.")
    print(
        f"Final model saved to: {final_path}"
    )


if __name__ == "__main__":
    train()