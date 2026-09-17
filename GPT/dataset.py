import json
import os
import requests
import pandas as pd
import pickle
import random

random.seed(42)

# FinQA dataset URL
url = "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/train.json"
file_path = "data/finqa_train.json"

if not os.path.exists(file_path):  # Check if the file already exists

    response = requests.get(url)  # Send request to download the file
    response.raise_for_status()  # Stop if the download failed

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(response.json(), f)  # Save the downloaded JSON data

    print("Download complete!")
else:
    print("FinQA already downloaded.")


with open(file_path, "r", encoding="utf-8") as f:
    finqa = json.load(f)  # Load the JSON data into Python

# Small test
finqa = finqa[:100]

print("Number of FinQA examples:", len(finqa))


def format_finqa_example(example):

    pre_text = " ".join(example["pre_text"])  # Combine the text before the table
    post_text = " ".join(example["post_text"])  # Combine the text after the table

    table = "\n".join(
        " | ".join(row)  # Join table cells with |
        for row in example["table"]
    )

    question = example["qa"]["question"]  # Get the question
    answer = example["qa"]["answer"]  # Get the answer

    text = (
        f"Context:\n"
        f"{pre_text}\n"
        f"{post_text}\n\n"
        f"Table:\n"
        f"{table}\n\n"
        f"Question:\n"
        f"{question}\n\n"
        f"Answer:\n"
        f"{answer}"
    )

    return {
        "text": text,
        "source": "finqa"  # Keep track of where the example came from
    }


dataset = [
    format_finqa_example(example)
    for example in finqa
]

print("Clean dataset size:", len(dataset))

print("\nFirst processed example:")
print(dataset[0]["text"])


# Financial PhraseBank dataset URL
url = "https://raw.githubusercontent.com/maxwellsarpong/NLP-financial-text-processing-dataset/master/Sentences_AllAgree.txt"
file_path = "data/phrasebank.txt"

if not os.path.exists(file_path):  # Check if the file already exists
    print("Downloading Financial PhraseBank...")

    response = requests.get(url)  # Download the dataset
    response.raise_for_status()  # Stop if the download failed

    with open(file_path, "wb") as f:
        f.write(response.content)  # Save the downloaded file

    print("Download complete!")
else:
    print("Financial PhraseBank already downloaded.")

with open(file_path, "r", encoding="latin-1") as f:
    lines = f.readlines()  # Read all lines from the file

# Small test
lines = lines[:100]

print("Number of PhraseBank examples:", len(lines))


def format_phrasebank_example(line):

    sentence, label = line.strip().rsplit("@", 1)  # Separate sentence from sentiment

    text = (
        f"Sentence:\n"
        f"{sentence}\n\n"
        f"Sentiment:\n"
        f"{label}"
    )

    return {
        "text": text,
        "source": "phrasebank"  # Keep track of the dataset source
    }


phrasebank_dataset = [
    format_phrasebank_example(line)
    for line in lines
]

print("Clean PhraseBank size:", len(phrasebank_dataset))

print("\nFirst PhraseBank example:")
print(phrasebank_dataset[0]["text"])


# FiQA dataset URL
url = "https://huggingface.co/datasets/vibrantlabsai/fiqa/resolve/main/data/main/train.parquet"
file_path = "data/fiqa_train.parquet"

if not os.path.exists(file_path):  # Check if the file already exists
    print("Downloading FiQA...")

    response = requests.get(url)  # Download the dataset
    response.raise_for_status()  # Stop if the download failed

    with open(file_path, "wb") as f:
        f.write(response.content)  # Save the downloaded file

    print("Download complete!")
else:
    print("FiQA already downloaded.")

fiqa = pd.read_parquet(file_path)  # Load the Parquet file into a DataFrame

# Small test
fiqa = fiqa.head(100)

print("Number of FiQA examples:", len(fiqa))
print("\nFiQA columns:")
print(fiqa.columns.tolist())  # Show the available columns


def format_fiqa_example(example):

    question = example["question"]  # Get the question
    answer = example["ground_truths"]  # Get the answer

    text = (
        f"Question:\n"
        f"{question}\n\n"
        f"Answer:\n"
        f"{answer}"
    )

    return {
        "text": text,
        "source": "fiqa"  # Keep track of the dataset source
    }


fiqa_dataset = [
    format_fiqa_example(row)
    for _, row in fiqa.iterrows()  # Go through each row
]

print("Clean FiQA size:", len(fiqa_dataset))

print("\nFirst FiQA example:")
print(fiqa_dataset[0]["text"])


# EDGAR dataset URL
url = "https://huggingface.co/datasets/JanosAudran/financial-reports-sec/resolve/refs%2Fconvert%2Fparquet/small_lite/train/0000.parquet"
file_path = "data/edgar_train.parquet"

if not os.path.exists(file_path):  # Check if the file already exists
    print("Downloading EDGAR...")

    response = requests.get(url)  # Download the dataset
    response.raise_for_status()  # Stop if the download failed

    with open(file_path, "wb") as f:
        f.write(response.content)  # Save the downloaded file

    print("Download complete!")
else:
    print("EDGAR already downloaded.")

edgar = pd.read_parquet(file_path)  # Load the Parquet file

# Small test
edgar = edgar.head(100)

print("Number of EDGAR examples:", len(edgar))
print("\nEDGAR columns:")
print(edgar.columns.tolist())  # Show the available columns


def format_edgar_example(example):

    sentence = example["sentence"]  # Get the financial report sentence

    text = (
        f"Financial Report:\n"
        f"{sentence}"
    )

    return {
        "text": text,
        "source": "edgar"  # Keep track of the dataset source
    }


edgar_dataset = [
    format_edgar_example(row)
    for _, row in edgar.iterrows()  # Go through each row
]

print("Clean EDGAR size:", len(edgar_dataset))

print("\nFirst EDGAR example:")
print(edgar_dataset[0]["text"])


# Combine all four datasets into one list
all_dataset = (
    dataset
    + phrasebank_dataset
    + fiqa_dataset
    + edgar_dataset
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
train_pkl_path = "data/ultrachat_train.pkl"
test_pkl_path = "data/ultrachat_test.pkl"

with open(train_pkl_path, "wb") as f:
    pickle.dump(train_dataset, f)
    print(f"\nTrain dataset saved to {train_pkl_path}")

with open(test_pkl_path, "wb") as f:
    pickle.dump(test_dataset, f)
    print(f"Test dataset saved to {test_pkl_path}")

print("\nDataset preparation complete!")