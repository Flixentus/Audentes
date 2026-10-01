import json
import os
import pickle
import random
import requests
import pandas as pd

from bpe_tokenizer import build_bpe_tokenizer

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

SEED = 42
random.seed(SEED)

CCNA_SIZE = 7500
NIT_SIZE = 7500
TOPOLOGY_SIZE = 5000
IOSXR_SIZE = 5000

TOKENIZER_PATH = os.path.join(DATA_DIR, "chat_bpe_tokenizer.json")
TRAIN_PKL = os.path.join(DATA_DIR, "ultrachat_train.pkl")
TEST_PKL = os.path.join(DATA_DIR, "ultrachat_test.pkl")

def download(url, path, name):
    if os.path.exists(path):
        print(f"{name} already exists.")
        return
    print(f"Downloading {name}...")
    r = requests.get(url)
    r.raise_for_status()
    with open(path, "wb") as f:
        f.write(r.content)
    print(f"{name} download complete!")

def make_example(question, answer, source):
    question = str(question).strip()
    answer = str(answer).strip()
    text = (
        "<bos>\n"
        "<|user|>\n"
        f"{question}\n"
        "<|assistant|>\n"
        f"{answer}\n"
        "<eos>"
    )
    return {
        "text": text,
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ],
        "source": source
    }

def save_pickle(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

ccna_path = os.path.join(DATA_DIR, "ccna_medium.parquet")
download(
    "https://huggingface.co/datasets/Rzkoohi/CCNA_medium/resolve/main/data/train-00000-of-00001.parquet",
    ccna_path, "CCNA"
)

ccna = pd.read_parquet(ccna_path, engine="fastparquet").dropna(subset=["question", "answer"])
ccna = ccna.sample(min(CCNA_SIZE, len(ccna)), random_state=SEED)
ccna_dataset = [
    make_example(row["question"], row["answer"], "ccna")
    for _, row in ccna.iterrows()
]
print(f"CCNA: {len(ccna_dataset)}")

nit_path = os.path.join(DATA_DIR, "nit.json")
download(
    "https://huggingface.co/datasets/Smarneh/NIT/resolve/main/NIT_dataset.json",
    nit_path, "NIT"
)

with open(nit_path, "r", encoding="utf-8") as f:
    nit = json.load(f)

nit = random.sample(nit, min(NIT_SIZE, len(nit)))
nit_dataset = [
    make_example(
        f"{x.get('question', '')}\n\nContext:\n{x.get('context', '')}",
        x.get("answer", ""),
        "nit"
    )
    for x in nit
]
print(f"NIT: {len(nit_dataset)}")

topology_path = os.path.join(DATA_DIR, "network_topology.csv")
download(
    "https://huggingface.co/datasets/Mohamed77777777777777777777777777/network-topology-troubleshooting-dataset/resolve/main/train.csv",
    topology_path, "Network Topology"
)

topology = pd.read_csv(topology_path).dropna(subset=["Question", "Response"])
topology = topology.sample(min(TOPOLOGY_SIZE, len(topology)), random_state=SEED)

topology_dataset = []
for _, row in topology.iterrows():
    reasoning = row.get("Reasoning", "")
    reasoning = "" if pd.isna(reasoning) else str(reasoning)
    answer = str(row["Response"]).strip()
    if reasoning:
        answer += f"\n\nReasoning:\n{reasoning}"
    topology_dataset.append(
        make_example(row["Question"], answer, "network_topology")
    )

print(f"Network Topology: {len(topology_dataset)}")

iosxr_path = os.path.join(DATA_DIR, "iosxr.parquet")
download(
    "https://huggingface.co/datasets/ramixpe/sp_llama_simple/resolve/main/data/train-00000-of-00001.parquet",
    iosxr_path, "Cisco IOS XR"
)

iosxr = pd.read_parquet(iosxr_path, engine="fastparquet").dropna(subset=["question", "answer"])
iosxr["question"] = iosxr["question"].astype(str)
iosxr["answer"] = iosxr["answer"].astype(str)
iosxr = iosxr[iosxr["answer"].str.len() > 20]
iosxr = iosxr[~iosxr["question"].str.match(r"^[A-Za-z0-9_]+-\d-[A-Z_]+$", na=False)]
iosxr = iosxr.sample(min(IOSXR_SIZE, len(iosxr)), random_state=SEED)

iosxr_dataset = [
    make_example(row["question"], row["answer"], "iosxr")
    for _, row in iosxr.iterrows()
]

print(f"Cisco IOS XR: {len(iosxr_dataset)}")

all_dataset = ccna_dataset + nit_dataset + topology_dataset + iosxr_dataset
random.shuffle(all_dataset)

split = int(len(all_dataset) * 0.8)
train_dataset = all_dataset[:split]
test_dataset = all_dataset[split:]

print(f"\nTotal: {len(all_dataset)}")
print(f"Train: {len(train_dataset)}")
print(f"Test: {len(test_dataset)}")

texts = [x["text"] for x in all_dataset]

tokenizer = build_bpe_tokenizer(
    texts,
    vocab_size=32000,
    save_path=TOKENIZER_PATH
)

hf_tokenizer = tokenizer.tok
vocab_size = hf_tokenizer.get_vocab_size()

print(f"Vocab size: {vocab_size}")

for token in ["<pad>", "<unk>", "<bos>", "<eos>", "<|user|>", "<|assistant|>"]:
    print(f"{token}: {hf_tokenizer.token_to_id(token)}")

if hf_tokenizer.token_to_id("<pad>") != 0:
    raise ValueError("<pad> must have token ID 0.")

save_pickle(train_dataset, TRAIN_PKL)
save_pickle(test_dataset, TEST_PKL)

with open(os.path.join(DATA_DIR, "train.jsonl"), "w", encoding="utf-8") as f:
    for x in train_dataset:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")

with open(os.path.join(DATA_DIR, "test.jsonl"), "w", encoding="utf-8") as f:
    for x in test_dataset:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")

print(f"\nSaved: {TRAIN_PKL}")
print(f"Saved: {TEST_PKL}")
print(f"Saved: {TOKENIZER_PATH}")
print("\nDataset preparation complete!")