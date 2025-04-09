from torch.utils.data import Dataset
import torch
from PIL import Image,ImageDraw,ImageFont
from uuid import uuid4
import random
import textwrap
from datasets import Dataset
from tqdm import tqdm
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath("../"))
current_dir = os.path.dirname(os.path.abspath(__file__))
tokenizer_path = os.path.join(current_dir, "../dataset/tokenizer.model")
tokenizer_path = os.path.abspath(tokenizer_path)

from tokenizer.tokenizer import Tokenizer

def prepare_image(image_path,text):

    img = Image.open(image_path).convert('RGB')

    width, height = img.size

    if width < 224 or height < 224:
        raise ValueError("Image is too small for a 224x224 crop!")

    left = random.randint(0, width - 224)
    top = random.randint(0, height - 224)
    right = left + 224
    bottom = top + 224
    img_cropped = img.crop((left, top, right, bottom))
    draw = ImageDraw.Draw(img_cropped)
    font_size = random.randint(15, 40)
    font = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSans.ttf", font_size)
    wrapper = textwrap.TextWrapper(width=30)
    wrapped_text = wrapper.wrap(text)

    line_spacing = 5
    line_height = draw.textbbox((0, 0), "Sample", font=font)[3] - draw.textbbox((0, 0), "Sample", font=font)[1]
    total_height = len(wrapped_text) * line_height + (len(wrapped_text) - 1) * line_spacing

    initial_font_size = font_size
    while total_height > 224 and font_size > 5:  # Minimum font size of 5 for readability
        font_size -= 1
        font = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSans.ttf", font_size)
        line_height = draw.textbbox((0, 0), "Sample", font=font)[3] - draw.textbbox((0, 0), "Sample", font=font)[1]
        total_height = len(wrapped_text) * line_height + (len(wrapped_text) - 1) * line_spacing

    print(f"Initial random font size: {initial_font_size}px, Adjusted font size: {font_size}px, Total text height: {total_height}px")

    max_x = 224 - max(draw.textbbox((0, 0), line, font=font)[2] - draw.textbbox((0, 0), line, font=font)[0] for line in wrapped_text)
    x_start = random.randint(0, max_x) if max_x > 0 else 0
    y_start = 0
    y = y_start
    for line in wrapped_text:
        draw.text((x_start, y), line, font=font, fill='black')
        y += line_height + line_spacing

    print(f"Adjusted font size: {font_size}px, Total text height: {total_height}px")
    image_name = f"{uuid4()}.png"
    img_cropped.save(image_name)
    return image_name

def prpare_data(csv_file,tokenize_path,image_list,image_directory):

    special_token = ["<en|gu>", "<gu|en>","<en>", "<gu>"]
    tokenizer = Tokenizer(special_token)
    tokenizer.load(tokenizer_path)
    data = pd.read_csv(csv_file, sep="\t")
    process_data=[]
    for i in tqdm(range(len(data))):
        random_image = random.choice(image_list)
        random_image1 = random.choice(image_list)
        row = data.iloc[i]
        src = row['src']
        tgt = row['tgt']

        if src is not str:
            src = str(src)
        if tgt is not str:
            tgt = str(tgt)

        src_image = prepare_image(random_image,src)
        tgt_image = prepare_image(random_image1,tgt)

        tokenized_src = tokenizer.encode(src)
        tokenized_tgt = tokenizer.encode(tgt)
        process_data.append({
            'src_lang': row['src_lang'],
            'tgt_lang': row['tgt_lang'],
            'src': tokenized_src,
            'tgt': tokenized_tgt,
            'scr_image':src_image,
            'tgt_image':tgt_image,
        })
    df_process_data = pd.DataFrame(process_data)
    data_set = Dataset.from_pandas(df_process_data)
    data_set.save_to_disk('eng_guj_img')


if __name__ == "__main__":
    prepare_image("test.png","Since you’re working with existing images and want to add text with a 15-pixel font size after taking a random 224x224 crop, I’ll provide a Python solution using Pillow to achieve this. I’ll assume you want to crop a random 224x224 section from the image, then add the text while keeping the final output within 224x224 pixels. For a manual approach, I’ll outline the steps as well. Well Elllllllll")
