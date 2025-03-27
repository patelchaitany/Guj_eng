from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split
import torch
from torch.nn.utils.rnn import pad_sequence
import sys
import os
from datasets import load_from_disk

sys.path.append(os.path.abspath("../"))
current_dir = os.path.dirname(os.path.abspath(__file__))  # Path to model/train.py
tokenizer_path = os.path.join(current_dir, "../dataset/tokenizer.model")
tokenizer_path = os.path.abspath(tokenizer_path)

from tokenizer.tokenizer import Tokenizer


def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_batch = pad_sequence(src_batch,batch_first=True,padding_value=3)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True,padding_value=3)

    return src_batch, tgt_batch


class TranslatorDataset(Dataset):
    def __init__(self, path, tokenizer_path):
        self.data = load_from_disk(path)
        self.special_token = ["<en|gu>", "<gu|en>","<en>", "<gu>"]
        self.tokenizer = Tokenizer(self.special_token)
        self.tokenizer.load(tokenizer_path)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        bos_token = self.tokenizer.get_token_id("bos")
        eos_token = self.tokenizer.get_token_id("eos")
        src_text = item["src"]
        tgt_text = item["tgt"]

        src_lang = item["src_lang"]
        tgt_lang = item["tgt_lang"]

        if src_lang == "eng_Latn":
            eng_token = self.tokenizer.get_token_id("<en|gu>")
            guj_token = self.tokenizer.get_token_id("<gu>")
            src_text = [eng_token] + src_text
            tgt_text = [guj_token] + tgt_text
        else:
            guj_token = self.tokenizer.get_token_id("<gu|en>")
            eng_token = self.tokenizer.get_token_id("<en>")
            src_text = [guj_token] + src_text
            tgt_text = [eng_token] + tgt_text

        src_text = [bos_token] + src_text + [eos_token]
        tgt_text = [bos_token] + tgt_text + [eos_token]

        return torch.tensor(src_text), torch.tensor(tgt_text)

if __name__ == "__main__":

    dataset = TranslatorDataset("eng_guj", tokenizer_path)
    data_loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

    special_token = ["<en|gu>", "<gu|en>","<en>", "<gu>"]
    tokenizer = Tokenizer(special_token)
    tokenizer.load(tokenizer_path)
    for src, tgt in data_loader:
        print(src)
        print(tgt)
        print(tokenizer.decode(src[0].tolist()))
        print(tokenizer.decode(tgt[0].tolist()))
        break
