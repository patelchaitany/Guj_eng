from json import load
from torch.utils.data import Dataset
from typing import Set
import os
import sys
import pandas as pd
from torch.nn.utils.rnn import pad_sequence

from datasets import load_from_disk
sys.path.append(os.path.abspath("../"))
current_dir = os.path.dirname(os.path.abspath(__file__))  # Path to model/train.py
tokenizer_path = os.path.join(current_dir, "../dataset/tokenizer.model")
tokenizer_path = os.path.abspath(tokenizer_path)

from tokenizer.tokenizer import Tokenizer

class ImageTextData(Dataset):
    '''
    This the Image and text Pair DataSet
    For The Given Image There is the Targate Lenguage Text

    Data Structre:
        Image : Source Lenguage Data Set
        Text : Target Lenguage Data Set

    How the data is stored in the File System:
        Imgae : Image is stored with the image_num.jpg
        Source Text : Source Lenguage Data In tokenize Form
        Target Text : Target Lenguage Data In tokenize Form
        scr_leng : source Lenguage -> ("eng_Latn")
        trg_leng : target Lenguage -> ("guj_Latn")
    '''

    def __init__(self,source_leng_path:str,target_leng_path:str,text_data:str,tokenizer_path:str):
        self.source_leng_path = source_leng_path
        self.target_leng_path = target_leng_path
        self.text_data = load_from_disk(text_data)

        self.special_token = ["<en|gu>", "<gu|en>","<en>", "<gu>"]
        self.tokenizer = Tokenizer(self.special_token)
        self.tokenizer.load(tokenizer_path)


    def __len__(self):
        return len(self.text_data)

    def __getitem__(self,idx):
        item = self.text_data[idx]

        bos_token = self.tokenizer.get_token_id("bos")
        eos_token = self.tokenizer.get_token_id("eos")

        src_lang = item["src_lang"]
        trg_lang = item["trg_lang"]

        src_text = item["src"]
        trg_text = item["trg"]

        pass
