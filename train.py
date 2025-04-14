import sys
import os
import math
from datasets import load_from_disk
from torch.optim.optimizer import Optimizer
from tqdm import tqdm


sys.path.append(os.path.abspath("../"))
current_dir = os.path.dirname(os.path.abspath(__file__))  # Path to model/train.py
tokenizer_path = os.path.join(current_dir, "../data/tokenizer.model")
tokenizer_path = os.path.abspath(tokenizer_path)

# from dataset.dataset import TranslatorDataset,collate_fn

from model.transformer import Transformer
from model.config import Config
from tokenizer.tokenizer import Tokenizer

from dataset.dataset import TranslatorDataset, collate_fn_text
from dataset.dataset_image import ImageDataset,collect_fn_image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
grad_accum = 4096//32
batch_size = 32
batch_size_val = 16
max_iter = 5
warmup_steps = 10000
max_lr = 6e-4
min_lr = 0.1*max_lr

grad_accum_img = 4096//32

img_batch = 32
img_batch_val = 2
print(f"Current directory: {current_dir} {tokenizer_path}")

tokenizer_path = "data/tokenizer.model"

dataset = TranslatorDataset("data/text_data/", tokenizer_path)
image_data = ImageDataset("data/eng_guj_img/",tokenizer_path,"data/images/")
# Calculate split sizes
total_size = len(dataset)
val_size = int(total_size * 0.2)
train_size = total_size - val_size

train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

img_val = int(len(image_data)*0.2)
img_train = len(image_data) - img_val

img_train_data,img_val_data = torch.utils.data.random_split(image_data,[img_train,img_val])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_text, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=False, collate_fn=collate_fn_text, num_workers=2)

img_t_loader = DataLoader(img_train_data,batch_size =img_batch,shuffle = True,collate_fn = collect_fn_image,num_workers=8)
img_v_loader = DataLoader(img_val_data,batch_size = img_batch_val,shuffle = False,collate_fn = collect_fn_image,num_workers = 2)


print(f"Training dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")

len_data = len(train_loader)
max_step = 50000
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
        scale = it / warmup_steps
        return (min_lr + scale*(max_lr -min_lr))/max_lr

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
os.makedirs(checkpoint_dir, exist_ok=True)  # Create checkpoint directory if it doesn't exist

model = Transformer(config).to(device)
checkpoint_path = os.path.join(checkpoint_dir, f"transformer_epoch_{2999}.pt")

# Check if checkpoint exists before loading
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print(f"Checkpoint found at {checkpoint_path}")
else:
    print(f"No checkpoint found at {checkpoint_path}, using initialized model")
    # Initialize an empty checkpoint to avoid errors
    checkpoint = {'model_state_dict': model.state_dict()}

# Load model weights
model.load_state_dict(checkpoint['model_state_dict'])

print(f"Model weights loaded from {checkpoint_path}")

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model loaded with {total_params:,} total parameters")
print(f"Model has {trainable_params:,} trainable parameters")


optimizer = optim.AdamW(model.parameters(), lr=4e-4,betas=(0.9, 0.95), eps=1e-8)

criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.get_token_id("pad"))

model.compile()
source_text = ["Hello, My name is jhon Doe and I am a data scientist."]

step = 0

for i in range(0,max_step):

    import time
    running_loss = 0.0
    t_loader = tqdm(train_loader, desc=f"Epoch {i}", leave=True, total=grad_accum)
    img_loader = tqdm(img_t_loader,desc = f"image Epocg {i}",leave = True,total = grad_accum_img)
    lr = get_lr(i)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


    # i need to change this validation function to image validation

    if (i+1)%3000 == 0 or i == max_step-1:
        test_loader = tqdm(val_loader, desc=f"Validation", leave=True)
        with torch.no_grad():
            for src, tgt in test_loader:
                src = src.to(device)
                tgt = tgt.to(device)
                output = model(tgt,src)
                output = output[:,:-1,:].contiguous()
                loss = criterion(output.view(-1, output.size(-1)), tgt[:,1:].contiguous().view(-1))
                test_loader.set_postfix(val_loss=loss.item())
                run.log({"val_loss":loss.item()})


    optimizer.zero_grad()
    for idx, (x, y) in enumerate(t_loader):
        if idx >= grad_accum:
            t_loader.close()
            break
        start_time = time.time()
        x = x.to(device)
        y = y.to(device)

        with torch.autocast(device_type="cuda",dtype=torch.float32):
            output = model(y,x)
            output = output[:,:-1,:].contiguous()
            loss = criterion(output.view(-1, output.size(-1)), y[:,1:].contiguous().view(-1))

        loss.backward()

        batch_time = time.time() - start_time
        running_loss += loss.item()

        # Update tqdm stats
        t_loader.set_postfix(
            loss=loss.item(),
            avg_loss=running_loss/(idx+1),
            batch_time=f"{batch_time:.4f}s"
        )

        run.log({"loss":loss.item(),"avg_loss":running_loss/(idx+1),"lr":lr})


    for idx,item in enumerate(img_loader):
        if idx >= grad_accum_img:
           img_loader.close()
           break

        start_time = time.time()
        image = item['image'].to(device)
        output = item['output'].to(device)

        with torch.autocast(device_type="cuda",dtype=torch.float32):
            pred = model(output,image,is_image = True)
            pred = pred[:,:-1,:].contiguous()
            loss = criterion(pred.view(-1,pred.size(-1)),output[:,1:].contiguous().view(-1))

        loss.backward()

        batch_time = time.time() - start_time
        running_loss += loss.item()

        img_loader.set_postfix(
            loss = loss.item(),
            avg_loss =running_loss/(idx+1),
            batch_time = f"{batch_time:.4f}s"
        )

        run.log({"loss_img":loss.item(),"avg_loss":running_loss/(idx+1),"lr":lr})
    optimizer.step()


    if i%100 == 0:
        validate_with_text(model,source_text,tokenizer,device)

    step += 1

    if((i+1)%3000 == 0 or i == max_step-1):
        # Save model checkpoint after each epoch
        checkpoint_dir = os.path.join(current_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_path = os.path.join(checkpoint_dir, f"transformer_epoch_{i}.pt")
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config
        }, checkpoint_path)

        print(f"Model saved to {checkpoint_path}")
