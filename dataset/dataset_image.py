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
import torchvision.transforms as transforms
from tokenizer.tokenizer import Tokenizer

def collect_fn_image(batch):
    images = torch.stack([item['image'] for item in batch])
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


def ocr_collate_fn(batch):
    images = torch.stack([item['image'] for item in batch])
    texts = [item['text'] for item in batch]

    input = pad_sequence(texts, batch_first=True, padding_value=3)
    return {
        'images': images,
        'input': input
    }

def translation_collate_fn(batch):
    images = torch.stack([item['image'] for item in batch])
    source_texts = [item['text'] for item in batch]
    target_texts = [item['text_gujarati'] for item in batch]

    input = pad_sequence(source_texts, batch_first=True, padding_value=3)
    output = pad_sequence(target_texts, batch_first=True, padding_value=3)
    return {
        'images': images,
        'input': input,
        'output': output
    }

class ImageDataset(Dataset):

    def __init__(self,path,tokenizer_path,image_folder):

        self.data = load_from_disk(path)
        self.special_token = ["<en|gu>", "<gu|en>","<en>", "<gu>"]
        self.tokenizer = Tokenizer(self.special_token)
        self.tokenizer.load(tokenizer_path)
        self.image_folder = image_folder
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
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
        # print("src lang ")
        if new_item['scr_lang'] == "eng_Latn":
            data = self.getguj(new_item)
        else:
            data = self.english(new_item)

        return data

    def getguj(self,item):

        # print("englisg 1 ",item['scr'])
        # print("gujrati 1 ",item['tgt'])
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

        # print("englisg ",item['scr'])
        # print("gujrati ",item['tgt'])
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

class OCRDataset(Dataset):
    def __init__(self, dataset_path,tokenizer_path):
        self.dataset = load_from_disk(dataset_path)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.special_token = ["<en|gu>", "<gu|en>","<en>", "<gu>"]
        self.tokenizer = Tokenizer(self.special_token)
        self.tokenizer.load(tokenizer_path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Convert PIL image to tensor and normalize
        image = item['image']
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        bos = self.tokenizer.get_token_id("bos")
        eos = self.tokenizer.get_token_id("eos")
        en = self.tokenizer.get_token_id("<en>")
        image_tensor = self.transform(image)
        text = item['text']
        text_tensor = self.tokenizer.encode(text)
        text_tensor = [bos] +[en] +text_tensor + [eos]
        text_tensor = torch.tensor(text_tensor)

        return {
            'image': image_tensor,
            'text': text_tensor
        }

class TranslationDataset(Dataset):
    def __init__(self, dataset_path, tokenizer_path):
        self.dataset = load_from_disk(dataset_path)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.special_token = ["<en|gu>", "<gu|en>","<en>", "<gu>"]
        self.tokenizer = Tokenizer(self.special_token)
        self.tokenizer.load(tokenizer_path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Convert PIL image to tensor and normalize
        image = item['image']
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        image_tensor = self.transform(image)

        bos = self.tokenizer.get_token_id("bos")
        eos = self.tokenizer.get_token_id("eos")
        gu = self.tokenizer.get_token_id("<en|gu>")
        text_tensor = self.tokenizer.encode(item['text'])
        text_tensor = [bos] + [gu] +text_tensor + [eos]
        text_tensor = torch.tensor(text_tensor)

        return {
            'image': image_tensor,
            'text': item['text'],
            'text_gujarati': item['text_gujarati']
        }


if __name__ == "__main__":

    current_dir = os.path.dirname(os.path.abspath(__file__))

    data_dir = os.path.join(os.path.dirname(current_dir),"data")
    dataset = ImageDataset(os.path.join(data_dir,"eng_guj_img"),tokenizer_path, os.path.join(data_dir,"images"))
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True,collate_fn = collect_fn)
    for data in data_loader:
        # print(" "*50)
        print(data['input'])
        print(data['output'])
