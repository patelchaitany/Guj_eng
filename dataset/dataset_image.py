from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split
import torch
from torch.nn.utils.rnn import pad_sequence
import sys
import os
from datasets import load_from_disk
import numpy as np
from PIL import Image
sys.path.append(os.path.abspath("../"))
current_dir = os.path.dirname(os.path.abspath(__file__))  # Path to model/train.py
tokenizer_path = os.path.join(current_dir, "../data/tokenizer.model")
tokenizer_path = os.path.abspath(tokenizer_path)

from tokenizer.tokenizer import Tokenizer

def collect_fn(batch):
    images = [item['image'] for item in batch]
    input =  [item['src_text'] for item in batch]
    output = [item['tgt_text'] for item in batch]
    original = [item['original'] for item in batch]
    input = pad_sequence(input, batch_first=True, padding_value=3)
    output = pad_sequence(output, batch_first=True, padding_value=3)
    original = pad_sequence(original, batch_first=True, padding_value=3)

    data = {
        'images': images,
        'input': input,
        'output': output,
        'original': original
    }
    return data

class ImageDataset(Dataset):

    def __init__(self,path,tokenizer_path,image_folder):

        self.data = load_from_disk(path)
        self.special_token = ["<en|gu>", "<gu|en>","<en>", "<gu>"]
        self.tokenizer = Tokenizer(self.special_token)
        self.tokenizer.load(tokenizer_path)
        self.image_folder = image_folder
    def __len__(self):
        return len(self.data)

    def __getitem__(self,index):

        item = self.data[index]
        random_number = np.random.uniform(low=0.0, high=1.0, size=1)[0]
        data = None
        new_item = {}
        if random_number > 0.5:
            new_item['scr'] = item['tgt']
            new_item['tgt'] = item['scr']
            new_item['scr_lang'] = item['tgt_lang']
            new_item['tgt_lang'] = item['scr_lang']
            new_item['scr_image'] = item['tgt_image']
            new_item['tgt_image'] = item['scr_image']

        else:
            new_item['scr'] = item['scr']
            new_item['tgt'] = item['tgt']
            new_item['scr_lang'] = item['scr_lang']
            new_item['tgt_lang'] = item['tgt_lang']
            new_item['scr_image'] = item['scr_image']
            new_item['tgt_image'] = item['tgt_image']

        # print("text src ",new_item['scr'])
        # print("text_tgt ",new_item['tgt'])

        if item['scr_lang'] == "eng_Latn":
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
        original_text = [english_token] + input_ids

        # print("token en ",src_text)
        # print("tgt eu ",tgt_text)
        src_text = [bos_token] + src_text + [eos_token]
        tgt_text = [bos_token] + tgt_text + [eos_token]
        original_text = [bos_token] + original_text + [eos_token]

        path = os.path.join(self.image_folder, item['scr_image'])

        image = Image.open(path)

        image = np.array(image)
        image = image.transpose((2, 0, 1))
        image = image.astype(np.float32) / 255.0

        image = torch.tensor(image)

        data = {
            'src_text': torch.tensor(src_text),
            'tgt_text': torch.tensor(tgt_text),
            'image': image,
            'original' : torch.tensor(original_text)
        }

        return data

    def english(self,item):

        # input : is gujarati
        # output : is english

        input_ids = self.tokenizer.encode(item['scr'])
        output_ids = self.tokenizer.encode(item['tgt'])


        bos_token = self.tokenizer.get_token_id("bos")
        eos_token = self.tokenizer.get_token_id("eos")

        guj_token = self.tokenizer.get_token_id("<gu|en>")
        eng_token = self.tokenizer.get_token_id("<en>")

        guj_token = self.tokenizer.get_token_id("<gu>")

        src_text = [eng_token] + input_ids
        tgt_text = [guj_token] + output_ids
        original_text = [guj_token] + input_ids
        print("token gn ",src_text)
        print("tgt gu ",tgt_text)
        src_text = [bos_token] + src_text + [eos_token]
        tgt_text = [bos_token] + tgt_text + [eos_token]
        original_text = [bos_token] + original_text + [eos_token]

        path = os.path.join(self.image_folder, item['scr_image'])

        image = Image.open(path)

        image = np.array(image)
        image = image.transpose((2, 0, 1))
        image = image.astype(np.float32) / 255.0

        image = torch.tensor(image)

        data = {
            'src_text': torch.tensor(src_text),
            'tgt_text': torch.tensor(tgt_text),
            'image': image,
            'original' : torch.tensor(original_text)
        }

        return data

if __name__ == "__main__":
    dataset = ImageDataset("eng_guj_img",tokenizer_path, "out_image")
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True,collate_fn = collect_fn)
    for data in data_loader:
        print(data['input'])
        print(data['output'])
