import json
import os
import requests
import pandas as pd
import random

random.seed(42)


# Ensure data directory exists
os.makedirs("data", exist_ok=True)


# CCNA dataset URL
url = "https://huggingface.co/datasets/Rzkoohi/CCNA_medium/resolve/main/data/train-00000-of-00001.parquet"
file_path = "data/ccna_medium.parquet"

if not os.path.exists(file_path):  # Check if the file already exists

    print("Downloading CCNA...")

    response = requests.get(url)  # Send request to download the file
    response.raise_for_status()  # Stop if the download failed

    with open(file_path, "wb") as f:
        f.write(response.content)  # Save the downloaded file

    print("Download complete!")
else:
    print("CCNA already downloaded.")


ccna = pd.read_parquet(file_path)  # Load the Parquet file

# Small test
ccna = ccna.head(100)

print("Number of CCNA examples:", len(ccna))

print("\nCCNA columns:")
print(ccna.columns.tolist())  # Show the available columns


def format_ccna_example(example):

    question = example["question"]  # Get the question
    answer = example["answer"]  # Get the answer

    # NEW:
    # Instead of putting everything into one plain "text" field,
    # we separate the user's question from the assistant's answer.
    return {
        "messages": [
            {
                "role": "user",
                "content": str(question)
            },
            {
                "role": "assistant",
                "content": str(answer)
            }
        ],
        "source": "ccna"  # Keep track of where the example came from
    }


ccna_dataset = [
    format_ccna_example(row)
    for _, row in ccna.iterrows()  # Go through each row
]

print("Clean CCNA size:", len(ccna_dataset))

print("\nFirst CCNA example:")
print(json.dumps(ccna_dataset[0], indent=2, ensure_ascii=False))


# NIT dataset URL
url = "https://huggingface.co/datasets/Smarneh/NIT/resolve/main/NIT_datset.json"
file_path = "data/nit.json"

if not os.path.exists(file_path):  # Check if the file already exists

    print("\nDownloading NIT...")

    response = requests.get(url)  # Download the dataset
    response.raise_for_status()  # Stop if the download failed

    with open(file_path, "wb") as f:
        f.write(response.content)  # Save the downloaded file

    print("Download complete!")
else:
    print("NIT already downloaded.")


with open(file_path, "r", encoding="utf-8") as f:
    nit = json.load(f)  # Load the downloaded JSON data

# Small test
nit = nit[:100]

print("Number of NIT examples:", len(nit))


def format_nit_example(example):

    question = example["question"]  # Get the question
    context = example["context"]  # Get the context
    answer = example["answer"]  # Get the answer

    # NEW:
    # The question and context become the user's input.
    # The answer becomes the assistant's response.
    user_message = (
        f"{question}\n\n"
        f"Context:\n"
        f"{context}"
    )

    return {
        "messages": [
            {
                "role": "user",
                "content": user_message
            },
            {
                "role": "assistant",
                "content": str(answer)
            }
        ],
        "source": "nit"  # Keep track of where the example came from
    }


nit_dataset = [
    format_nit_example(example)
    for example in nit
]

print("Clean NIT size:", len(nit_dataset))

print("\nFirst NIT example:")
print(json.dumps(nit_dataset[0], indent=2, ensure_ascii=False))


# Combine all datasets into one list
all_dataset = (
    ccna_dataset
    + nit_dataset
)

print("Total SFT dataset size:", len(all_dataset))


# Shuffle the combined dataset
random.shuffle(all_dataset)


# 80/20 train/test split
split_idx = int(0.8 * len(all_dataset))

train_dataset = all_dataset[:split_idx]
test_dataset = all_dataset[split_idx:]


print(f"\nTrain set size: {len(train_dataset)}")
print(f"Test set size: {len(test_dataset)}")


# NEW:
# Save as JSONL instead of pickle.
# JSONL makes it easy for us and our teammates to inspect
# individual examples and use them in the MoE training pipeline.
train_path = "data/train.jsonl"
test_path = "data/test.jsonl"


with open(train_path, "w", encoding="utf-8") as f:
    for example in train_dataset:
        f.write(json.dumps(example, ensure_ascii=False) + "\n")

print(f"\nTrain dataset saved to {train_path}")


with open(test_path, "w", encoding="utf-8") as f:
    for example in test_dataset:
        f.write(json.dumps(example, ensure_ascii=False) + "\n")

print(f"Test dataset saved to {test_path}")


print("\nDataset preparation complete!")