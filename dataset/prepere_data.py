import sys
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np
from datasets import Dataset

sys.path.append(os.path.abspath("../"))
current_dir = os.path.dirname(os.path.abspath(__file__))  # Path to model/train.py
tokenizer_path = os.path.join(current_dir, "../dataset/tokenizer.model")
tokenizer_path = os.path.abspath(tokenizer_path)

from tokenizer.tokenizer import Tokenizer
# prepering the data as numpy shard file with pretokenization with all the token added

def prepare_data(csv_file,tokenizer_path):
    special_token = ["<en|gu>", "<gu|en>","<en>", "<gu>"]
    # tokenizer = Tokenizer(special_token)
    # tokenizer.load(tokenizer_path)
    data = pd.read_csv(csv_file, sep="\t")
    # data = pd.read_csv(csv_file)
    process_data=[]
    for i in tqdm(range(len(data))):
        row = data.iloc[i]
        src = row['src']
        tgt = row['tgt']
        if src is not str:
            src = str(src)
        if tgt is not str:
            tgt = str(tgt)
        # tokenized_src = tokenizer.encode(src)
        # tokenized_tgt = tokenizer.encode(tgt)
        process_data.append({
            'src_lang': row['src_lang'],
            'tgt_lang': row['tgt_lang'],
            'src': src,
            'tgt': tgt
        })
    df_process_data = pd.DataFrame(process_data)
    data_set = Dataset.from_pandas(df_process_data)
    data_set.save_to_disk('text_data')
if __name__ == "__main__":
    prepare_data("refined.tsv",tokenizer_path)

    # prepare_data("tep.csv",tokenizer_path)
