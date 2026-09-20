import json
import os
import requests
import pandas as pd
import pickle
import random

random.seed(42)


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

print("Number of CCNA examples:", len(ccna))

print("\nCCNA columns:")
print(ccna.columns.tolist())  # Show the available columns


def format_ccna_example(example):

    question = example["question"]  # Get the question
    answer = example["answer"]  # Get the answer

    return {
        "messages": [
            {
                "role": "user",
                "content": question
            },
            {
                "role": "assistant",
                "content": answer
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
print(ccna_dataset[0])


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

print("Number of NIT examples:", len(nit))


def format_nit_example(example):

    question = example["question"]  # Get the question
    context = example["context"]  # Get the context
    answer = example["answer"]  # Get the answer

    text = (
        f"{question}\n\n"
        f"Context:\n"
        f"{context}"
    )

    return {
        "messages": [
            {
                "role": "user",
                "content": text
            },
            {
                "role": "assistant",
                "content": answer
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
print(nit_dataset[0])


# Combine all datasets into one list
all_dataset = (
    ccna_dataset
    + nit_dataset
)

print("Total dataset size:", len(all_dataset))


# Shuffle the combined dataset
random.shuffle(all_dataset)


# 80/20 train/test split
split_idx = int(0.8 * len(all_dataset))
train_dataset = all_dataset[:split_idx]
test_dataset = all_dataset[split_idx:]


print(f"\nTrain set size: {len(train_dataset)}")
print(f"Test set size: {len(test_dataset)}")


# Ensure data directory exists
os.makedirs("data", exist_ok=True)


# Save train and test datasets as pickle files
train_pkl_path = "data/network_train.pkl"
test_pkl_path = "data/network_test.pkl"


with open(train_pkl_path, "wb") as f:
    pickle.dump(train_dataset, f)
    print(f"\nTrain dataset saved to {train_pkl_path}")


with open(test_pkl_path, "wb") as f:
    pickle.dump(test_dataset, f)
    print(f"Test dataset saved to {test_pkl_path}")


print("\nDataset preparation complete!")
