from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split
import torch
from torch.nn.utils.rnn import pad_sequence
import sys
import os
from datasets import load_from_disk
import numpy as np

sys.path.append(os.path.abspath("../"))
current_dir = os.path.dirname(os.path.abspath(__file__))  # Path to model/train.py
tokenizer_path = os.path.join(current_dir, "../data/tokenizer.model")
tokenizer_path = os.path.abspath(tokenizer_path)

from tokenizer.tokenizer import Tokenizer


def collate_fn_text(batch):
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

        random_number = np.random.uniform(low=0.0, high=1.0, size=1)[0]
        data = None
        new_item = {}

        if random_number > 0.5:
            new_item['scr'] = item['tgt']
            new_item['tgt'] = item['src']
            new_item['src_lang'] = item['tgt_lang']
            new_item['tgt_lang'] = item['src_lang']
        else:
            new_item['scr'] = item['src']
            new_item['tgt'] = item['tgt']
            new_item['src_lang'] = item['src_lang']
            new_item['tgt_lang'] = item['tgt_lang']

        src_lang = new_item["src_lang"]
        tgt_lang = new_item["tgt_lang"]

        if src_lang == "eng_Latn":
            data = self.getguj(new_item)
        else:
            data = self.english(new_item)

        return data

    def getguj(self,item):
        input_ids = self.tokenizer.encode(item['scr'])
        output_ids = self.tokenizer.encode(item['tgt'])

        bos_token = self.tokenizer.get_token_id("bos")
        eos_token = self.tokenizer.get_token_id("eos")
        pad_token = self.tokenizer.get_token_id("<pad>")
        eng_token = self.tokenizer.get_token_id("<en|gu>")
        guj_token = self.tokenizer.get_token_id("<gu>")

        # print("pad token ",pad_token)

        english_token = self.tokenizer.get_token_id("<en>")

        src_text = [eng_token] + input_ids
        tgt_text = [guj_token] + output_ids

        # print("token en ",src_text)
        # print("tgt eu ",tgt_text)
        src_text = [bos_token] + src_text + [eos_token]
        tgt_text = [bos_token] + tgt_text + [eos_token]
        return torch.tensor(src_text), torch.tensor(tgt_text)

    def english(self,item):

        # input : is gujarati
        # output : is english

        input_ids = self.tokenizer.encode(item['scr'])
        output_ids = self.tokenizer.encode(item['tgt'])


        bos_token = self.tokenizer.get_token_id("bos")
        eos_token = self.tokenizer.get_token_id("eos")

        guj_token = self.tokenizer.get_token_id("<gu|en>")
        eng_token = self.tokenizer.get_token_id("<en>")

        # guj_token = self.tokenizer.get_token_id("<gu>")

        src_text = [guj_token] + input_ids
        tgt_text = [eng_token] + output_ids

        src_text = [bos_token] + src_text + [eos_token]
        tgt_text = [bos_token] + tgt_text + [eos_token]


        return torch.tensor(src_text), torch.tensor(tgt_text)


if __name__ == "__main__":

    data_dir = os.path.join(os.path.dirname(current_dir),"data")
    text_data = os.path.join(data_dir,"text_data")
    dataset = TranslatorDataset(text_data, tokenizer_path)
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
