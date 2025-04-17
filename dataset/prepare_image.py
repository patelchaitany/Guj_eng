from torch.utils.data import Dataset
from PIL import Image, ImageDraw, ImageFont
from uuid import uuid4
import random
import textwrap
from datasets import Dataset
from tqdm import tqdm
import pandas as pd
import sys
import os
import numpy as np
import cv2
from multiprocessing import Pool, cpu_count
import argparse

sys.path.append(os.path.abspath("../"))
current_dir = os.path.dirname(os.path.abspath(__file__))
tokenizer_path = os.path.join(current_dir, "../data/tokenizer.model")
tokenizer_path = os.path.abspath(tokenizer_path)

from tokenizer.tokenizer import Tokenizer  

def prepare_image(image_path, text, path, leng):
    img = Image.open(image_path).convert('RGB')
    width, height = img.size

    if width < 224 or height < 224:
        raise ValueError("Image is too small for a 224x224 crop!")

    left = random.randint(0, width - 224)
    top = random.randint(0, height - 224)
    img_cropped = img.crop((left, top, left + 224, top + 224))
    draw = ImageDraw.Draw(img_cropped)
    font_size = 15

    if leng == "en":
        font = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSans.ttf", font_size)
    else:
        font = ImageFont.truetype("/teamspace/studios/this_studio/ml_impl/font/NotoSansGujarati-Black.ttf", font_size)

    wrapper = textwrap.TextWrapper(width=30)
    wrapped_text = wrapper.wrap(text)

    line_spacing = 5
    line_height = draw.textbbox((0, 0), "Sample", font=font)[3] - draw.textbbox((0, 0), "Sample", font=font)[1]
    total_height = len(wrapped_text) * line_height + (len(wrapped_text) - 1) * line_spacing

    while total_height > 224 and font_size > 5:
        font_size -= 1
        if leng == "en":
            font = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSans.ttf", font_size)
        else:
            font = ImageFont.truetype("/teamspace/studios/this_studio/ml_impl/font/NotoSansGujarati-Black.ttf", font_size)
        line_height = draw.textbbox((0, 0), "Sample", font=font)[3] - draw.textbbox((0, 0), "Sample", font=font)[1]
        total_height = len(wrapped_text) * line_height + (len(wrapped_text) - 1) * line_spacing

    max_text_width = max(draw.textbbox((0, 0), line, font=font)[2] for line in wrapped_text)
    x_start = random.randint(0, 224 - max_text_width) if (224 - max_text_width) > 0 else 0
    y_start = 0

    text_area = img_cropped.crop((x_start, y_start, x_start + max_text_width, y_start + total_height))
    rgb_array = np.array(text_area)

    contrast_color = []
    for channel in range(3):
        gray = rgb_array[:, :, channel]
        _, thr_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        num_zeros = np.sum(thr_image == 0)
        num_ones = np.sum(thr_image == 255)
        thr_cnt = num_ones / (num_zeros + num_ones)
        channel_value = 0 if thr_cnt > 0.5 else 255
        contrast_color.append(channel_value)

    text_color = tuple(contrast_color)

    y = y_start
    for line in wrapped_text:
        draw.text((x_start, y), line, font=font, fill=text_color)
        y += line_height + line_spacing

    image_name = f"{uuid4()}.png"
    save_path = os.path.join(path, image_name)
    img_cropped.save(save_path)
    return image_name

def process_row(args):
    row, image_list, image_directory = args
    random_image = random.choice(image_list)
    random_image1 = random.choice(image_list)

    src = row['src']
    tgt = row['tgt']

    if src is not str:
        src = str(src)
    if tgt is not str:
        tgt = str(tgt)

    su = "gu"
    tg = "en"
    if row['src_lang'] == "eng_Latn":
        su = "en"
        tg = "gu"

    src_image = prepare_image(random_image, src, image_directory, su)
    tgt_image = prepare_image(random_image1, tgt, image_directory, tg)

    return {
        'scr_lang': row['src_lang'],
        'tgt_lang': row['tgt_lang'],
        'scr': src,
        'tgt': tgt,
        'scr_image': src_image,
        'tgt_image': tgt_image,
    }

def prpare_data(csv_file, image_list, image_directory, data_dir, num_rows=None, num_workers=None):
    data = pd.read_csv(csv_file, sep="\t")

    if num_rows is not None:
        data = data.head(num_rows)

    print(f"Processing {len(data)} rows using multiprocessing...")

    args = [(data.iloc[i], image_list, image_directory) for i in range(len(data))]

    with Pool(processes=num_workers or cpu_count()) as pool:
        process_data = list(tqdm(pool.imap(process_row, args), total=len(args)))

    df_process_data = pd.DataFrame(process_data)
    data_set = Dataset.from_pandas(df_process_data)
    out_path = os.path.join(data_dir, "eng_guj_img")
    data_set.save_to_disk(out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_rows", type=int, default=None, help="Number of rows to process")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes")
    args = parser.parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(current_dir), "data")
    output_image_dir = os.path.join(data_dir, "images")
    os.makedirs(output_image_dir, exist_ok=True)

    image_dir = os.path.join(os.path.dirname(current_dir), "image_data")
    image_files = []

    if os.path.exists(image_dir):
        for filename in os.listdir(image_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                image_files.append(os.path.join(image_dir, filename))
    else:
        print(f"Directory '{image_dir}' not found")

    prpare_data("refined.tsv", image_files, output_image_dir, data_dir, args.num_rows, args.workers)
