from tokenizers import Tokenizer as HFTokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
import os

class BPETokenizerWrapper():
    def __init__(self, hf_tokenizer):
        self.tok = hf_tokenizer
        self.word_to_idx = hf_tokenizer.get_vocab()
        self.idx_to_word = {v: k for k, v in self.word_to_idx.items()}

    def __len__(self):
        return self.tok.get_vocab_size()

    def encode(self, text):
        return self.tok.encode(
            text,
            add_special_tokens=False
        ).ids

    def decode(self, ids):
        special_ids = {self.word_to_idx[t] for t in ["<pad>", "<bos>", "<eos>", "<unk>"]}
        filtered = [i for i in ids if i not in special_ids]
        return self.tok.decode(filtered)

def build_bpe_tokenizer(combined_texts, vocab_size=32000, save_path="bpe_tokenizer.json"):
    if os.path.exists(save_path):
        tokenizer_bpe = HFTokenizer.from_file(save_path)
        
    else:
        tokenizer_bpe = HFTokenizer(BPE(unk_token="<unk>"))
        tokenizer_bpe.pre_tokenizer = ByteLevel()
        tokenizer_bpe.decoder = ByteLevelDecoder()

        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["<pad>", "<unk>", "<bos>", "<eos>", "<|user|>", "<|assistant|>"]
        )

        tokenizer_bpe.train_from_iterator(combined_texts, trainer=trainer)
        tokenizer_bpe.save(save_path)

    return BPETokenizerWrapper(tokenizer_bpe)