import sys
import os
import math
from datasets import load_from_disk
from torch.optim.optimizer import Optimizer
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import Normalize
from collections import defaultdict

# [1] 26454

sys.path.append(os.path.abspath("../"))
current_dir = os.path.dirname(os.path.abspath(__file__))  # Path to model/train.py
tokenizer_path = os.path.join(current_dir, "../data/tokenizer.model")
tokenizer_path = os.path.abspath(tokenizer_path)

# from dataset.dataset import TranslatorDataset,collate_fn

from model.transformer import Transformer
from model.config import Config
from tokenizer.tokenizer import Tokenizer

from dataset.dataset import TranslatorDataset, collate_fn_text
from dataset.dataset_image import ImageDataset,OCRDataset,TranslationDataset,collect_fn_image,translation_collate_fn,ocr_collate_fn
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
grad_accum = 4096//128
batch_size = 128
batch_size_val = 8
max_iter = 5
warmup_steps = 1000
max_lr = 6e-4
min_lr = 0.1*max_lr

ocr_lmbd = 0.5
mt_lmbd = 0.5
tit_lmbd = 1.2

grad_accum_img = 4096//16

img_batch = 16
img_batch_val = 2
print(f"Current directory: {current_dir} {tokenizer_path}")

tokenizer_path = "data/tokenizer.model"

dataset = TranslatorDataset("data/text_data/", tokenizer_path)
image_data = ImageDataset("data/eng_guj_img/",tokenizer_path,"data/images/")
ocr_data = OCRDataset("data/handwritten_text_ocr_224x224",tokenizer_path)
validation_data = TranslationDataset("data/handwritten_text_translation_validation_224x224",tokenizer_path)
total_size = len(dataset)
val_size = int(total_size * 0.2)
train_size = total_size - val_size

train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

img_val = int(len(image_data)*0.2)
img_train = len(image_data) - img_val

img_train_data,img_val_data = torch.utils.data.random_split(image_data,[img_train,img_val])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_text, num_workers=16)
# val_loader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=False, collate_fn=collate_fn_text, num_workers=2)

img_t_loader = DataLoader(img_train_data,batch_size =img_batch,shuffle = True,collate_fn = collect_fn_image,num_workers=16)
# img_v_loader = DataLoader(img_val_data,batch_size = img_batch_val,shuffle = False,collate_fn = collect_fn_image,num_workers = 2)

ocr_data_t = DataLoader(ocr_data,batch_size = img_batch,shuffle = True,collate_fn = ocr_collate_fn,num_workers=16)

valid_loader = DataLoader(validation_data,batch_size=2,shuffle=False,collate_fn =translation_collate_fn,num_workers = 8)

print(f"Training dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")

len_data = len(train_loader)
max_step = 10000
print(f"Total steps: {max_step}")
def validate_with_text(model, source_texts, tokenizer, device, max_len=50):
    print("\nValidation with provided text:")

    with torch.no_grad():

        bos_token = tokenizer.get_token_id("bos")
        eos_token = tokenizer.get_token_id("eos")
        for src_text in source_texts:
            src_ids = tokenizer.encode(src_text)

            eng_token = tokenizer.get_token_id("<en|gu>")
            src_ids = [eng_token] + [bos_token] + src_ids + [eos_token]
            src_tensor = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(device)

            guj_token = tokenizer.get_token_id("<gu>")
            start_token_id = bos_token or 1  # Fallback to 1 if not found
            tgt_tensor = torch.tensor([[guj_token,start_token_id]], dtype=torch.long).to(device)  # (1, 1)

            generated_ids = []
            for _ in range(max_len):
                output = model(tgt_tensor,src_tensor)  # (1, tgt_len, vocab_size)
                next_token = output[:, -1, :].argmax(dim=-1).item()  # Predict next token
                if next_token == eos_token:  # Stop at <END>
                    break
                generated_ids.append(next_token)
                tgt_tensor = torch.cat([tgt_tensor, torch.tensor([[next_token]], device=device)], dim=1)

            src_decoded = tokenizer.decode(src_ids)
            pred_decoded = tokenizer.decode(generated_ids)

            print(f"Source: {src_decoded}")
            print(f"Generated: {pred_decoded}\n")



def get_lr(it):
    if it<=warmup_steps:
        scale = (it+1) / warmup_steps
        return max_lr * scale

    if it>max_step:
        return min_lr

    deacy_ration = (it-warmup_steps)/(max_step-warmup_steps)
    assert 0<=deacy_ration<=1
    return min_lr + 0.5*(max_lr-min_lr)*(1+torch.cos(torch.tensor(deacy_ration)*math.pi))



special_token = ["<en|gu>", "<gu|en>","<en>", "<gu>"]
tokenizer_path = "data/tokenizer.model"
tokenizer = Tokenizer(special_token)
tokenizer.load(tokenizer_path)

vocab_size = tokenizer.get_vocab_size()

config = Config(
    vocab_size = vocab_size
)

run = wandb.init(project="final_run",config=config)
print(config)



print(len(train_loader))
checkpoint_dir = os.path.join(current_dir, "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)  

model = Transformer(config).to(device)
checkpoint_path = os.path.join(checkpoint_dir, f"transformer_epoch_{2099}.pt")

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print(f"Checkpoint found at {checkpoint_path}")
else:
    print(f"No checkpoint found at {checkpoint_path}, using initialized model")
    checkpoint = {'model_state_dict': model.state_dict()}

model.load_state_dict(checkpoint['model_state_dict'])

print(f"Model weights loaded from {checkpoint_path}")

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model loaded with {total_params:,} total parameters")
print(f"Model has {trainable_params:,} trainable parameters")


optimizer = optim.AdamW(model.parameters(), lr=4e-4,betas=(0.9, 0.95), eps=1e-8)

if 'optimizer_state_dict' in checkpoint:
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.get_token_id("pad"))

model.compile()
source_text = ["Hello, My name is jhon Doe and I am a data scientist."]


step = 0

for i in range(0, max_step):
    import time

    t_loader = tqdm(train_loader, desc=f"Epoch {i}", leave=True, total=grad_accum)
    img_loader = tqdm(img_t_loader, desc=f"Image Epoch {i}", leave=True, total=grad_accum_img)
    ocr_loader = tqdm(ocr_data_t, desc=f"OCR Epoch {i}", leave=True, total=grad_accum_img)
    lr = get_lr(i)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # === Validation ===
    if (i + 1) % 200 == 0 or i == max_step - 1:
        test_loader = tqdm(valid_loader, desc="Validation", leave=True)
        with torch.no_grad():
            for item in test_loader:
                src = item['images'].to(device)
                tgt = item['output'].to(device)
                output = model(tgt, src, is_image=True)
                output = output[:, :-1, :].contiguous()
                loss = criterion(output.view(-1, output.size(-1)), tgt[:, 1:].contiguous().view(-1))
                test_loader.set_postfix(val_loss=loss.item())
                run.log({"val_loss": loss.item()})

    # === MT Task ===
    optimizer.zero_grad()
    last_mt_loss = 0.0
    for idx, (x, y) in enumerate(t_loader):
        if idx >= grad_accum:
            t_loader.close()
            break

        x = x.to(device)
        y = y.to(device)
        start_time = time.time()

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(y, x)
            output = output[:, :-1, :].contiguous()
            loss = mt_lmbd * criterion(output.view(-1, output.size(-1)), y[:, 1:].contiguous().view(-1))
            last_mt_loss = loss.item()

        loss.backward()
        t_loader.set_postfix(loss=loss.item(), batch_time=f"{time.time() - start_time:.4f}s")

    # === TIT Task ===
    last_tit_loss = 0.0
    for idx, item in enumerate(img_loader):
        if idx >= grad_accum_img:
            img_loader.close()
            break

        image = item['images'].to(device)
        output = item['output'].to(device)
        start_time = time.time()

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = model(output, image, is_image=True)
            pred = pred[:, :-1, :].contiguous()
            loss = tit_lmbd * criterion(pred.view(-1, pred.size(-1)), output[:, 1:].contiguous().view(-1))
            last_tit_loss = loss.item()

        loss.backward()
        img_loader.set_postfix(loss=loss.item(), batch_time=f"{time.time() - start_time:.4f}s")

    # === OCR Task ===
    last_ocr_loss = 0.0
    for idx, item in enumerate(ocr_loader):
        if idx >= grad_accum_img:
            ocr_loader.close()
            break

        image = item['images'].to(device)
        output = item['input'].to(device)
        start_time = time.time()

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = model(output, image, is_image=True)
            pred = pred[:, :-1, :].contiguous()
            loss = ocr_lmbd * criterion(pred.view(-1, pred.size(-1)), output[:, 1:].contiguous().view(-1))
            last_ocr_loss = loss.item()

        loss.backward()
        ocr_loader.set_postfix(loss=loss.item(), batch_time=f"{time.time() - start_time:.4f}s")

    optimizer.step()
    total_loss = last_mt_loss + last_tit_loss + last_ocr_loss
    run.log({"total_loss": total_loss, "lr": lr})

    if i % 100 == 0:
        validate_with_text(model, source_text, tokenizer, device)

    step += 1
    if (i + 1) % 50 == 0 or i == max_step - 1:
        checkpoint_dir = os.path.join(current_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"transformer_epoch_{i}.pt")
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config
        }, checkpoint_path)
        print(f"Model saved to {checkpoint_path}")

