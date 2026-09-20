import json
import os
import requests
import pandas as pd
import random

random.seed(42)


# Ensure data directory exists
os.makedirs("data", exist_ok=True)


# CCNA dataset
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


ccna = pd.read_parquet(file_path)  # Load the dataset

# Small test
ccna = ccna.head(100)

print("Number of CCNA examples:", len(ccna))

print("\nCCNA columns:")
print(ccna.columns.tolist())


def format_ccna_example(example):

    question = example["question"]
    answer = example["answer"]

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
        "source": "ccna"
    }


ccna_dataset = [
    format_ccna_example(row)
    for _, row in ccna.iterrows()
]

print("Clean CCNA size:", len(ccna_dataset))

print("\nFirst CCNA example:")
print(json.dumps(ccna_dataset[0], indent=2, ensure_ascii=False))


# NIT dataset
url = "https://huggingface.co/datasets/Smarneh/NIT/resolve/main/NIT_datset.json"
file_path = "data/nit.json"

if not os.path.exists(file_path):

    print("\nDownloading NIT...")

    response = requests.get(url)
    response.raise_for_status()

    with open(file_path, "wb") as f:
        f.write(response.content)

    print("Download complete!")
else:
    print("NIT already downloaded.")


with open(file_path, "r", encoding="utf-8") as f:
    nit = json.load(f)

# Small test
nit = nit[:100]

print("Number of NIT examples:", len(nit))


def format_nit_example(example):

    question = example["question"]
    context = example["context"]
    answer = example["answer"]

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
        "source": "nit"
    }


nit_dataset = [
    format_nit_example(example)
    for example in nit
]

print("Clean NIT size:", len(nit_dataset))

print("\nFirst NIT example:")
print(json.dumps(nit_dataset[0], indent=2, ensure_ascii=False))


# Network Topology Troubleshooting dataset
url = "https://huggingface.co/datasets/Mohamed77777777777777777777777777/network-topology-troubleshooting-dataset/resolve/main/train.csv"
file_path = "data/network_topology.csv"

if not os.path.exists(file_path):

    print("\nDownloading Network Topology dataset...")

    response = requests.get(url)
    response.raise_for_status()

    with open(file_path, "wb") as f:
        f.write(response.content)

    print("Download complete!")
else:
    print("Network Topology dataset already downloaded.")


topology = pd.read_csv(file_path)

# Small test
topology = topology.head(100)

print("Number of Topology examples:", len(topology))

print("\nTopology columns:")
print(topology.columns.tolist())


def format_topology_example(example):

    question = example["Question"]
    response = example["Response"]
    reasoning = example["Reasoning"]

    answer = (
        f"{response}\n\n"
        f"Reasoning:\n"
        f"{reasoning}"
    )

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
        "source": "network_topology"
    }


topology_dataset = [
    format_topology_example(row)
    for _, row in topology.iterrows()
]

print("Clean Topology size:", len(topology_dataset))

print("\nFirst Topology example:")
print(json.dumps(topology_dataset[0], indent=2, ensure_ascii=False))


# DDoS Security dataset
url = "https://huggingface.co/datasets/sudiptob2/ddos-qna-dataset/resolve/main/data/train-00000-of-00001.parquet"
file_path = "data/ddos_qna.parquet"

if not os.path.exists(file_path):

    print("\nDownloading DDoS dataset...")

    response = requests.get(url)
    response.raise_for_status()

    with open(file_path, "wb") as f:
        f.write(response.content)

    print("Download complete!")
else:
    print("DDoS dataset already downloaded.")


ddos = pd.read_parquet(file_path)

# Small test
ddos = ddos.head(100)

print("Number of DDoS examples:", len(ddos))

print("\nDDoS columns:")
print(ddos.columns.tolist())


def format_ddos_example(example):

    question = example["title"]
    answer = example["text"]

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
        "source": "ddos_security"
    }


ddos_dataset = [
    format_ddos_example(row)
    for _, row in ddos.iterrows()
]

print("Clean DDoS size:", len(ddos_dataset))

print("\nFirst DDoS example:")
print(json.dumps(ddos_dataset[0], indent=2, ensure_ascii=False))


# Combine all datasets into one list
all_dataset = (
    ccna_dataset
    + nit_dataset
    + topology_dataset
    + ddos_dataset
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


# Save as JSONL
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

