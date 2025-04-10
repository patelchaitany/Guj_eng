import os
import random
import textwrap
import shutil
import csv
import cv2
from PIL import Image, ImageDraw, ImageFont
from datasets import load_dataset
from uuid import uuid4
from tqdm import tqdm
import numpy as np

BACKGROUND_FOLDER = "background_images"
ORIGINAL_FOLDER = "original_images"
SYNTHETIC_FOLDER = "synthetic_images"
PAIR_FILE = "pairs.csv"
FONT_PATH = "C:\\Windows\\Fonts\\arial.ttf"
NUM_IMAGES = 1000

os.makedirs(ORIGINAL_FOLDER, exist_ok=True)
os.makedirs(SYNTHETIC_FOLDER, exist_ok=True)

dataset = load_dataset("toghrultahirov/handwritten_text_ocr", split="train")

backgrounds = [os.path.join(BACKGROUND_FOLDER, f)
               for f in os.listdir(BACKGROUND_FOLDER)
               if f.lower().endswith((".png", ".jpg", ".jpeg"))]
if not backgrounds:
    raise ValueError("No background images found!")



import numpy as np

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import random
import textwrap
from uuid import uuid4

def create_synthetic_image(background_path, text, output_path):
    img = Image.open(background_path).convert('RGB')
    width, height = img.size

    if width < 224 or height < 224:
        raise ValueError("Image is too small for a 224x224 crop!")

    left = random.randint(0, width - 224)
    top = random.randint(0, height - 224)
    img_cropped = img.crop((left, top, left + 224, top + 224))

    draw = ImageDraw.Draw(img_cropped)
    font_size = 15
    font = ImageFont.truetype(FONT_PATH, font_size)

    wrapper = textwrap.TextWrapper(width=30)
    wrapped_text = wrapper.wrap(text)

    line_spacing = 5
    line_height = draw.textbbox((0, 0), "Sample", font=font)[3] - draw.textbbox((0, 0), "Sample", font=font)[1]
    total_height = len(wrapped_text) * line_height + (len(wrapped_text) - 1) * line_spacing

    while total_height > 224 and font_size > 5:
        font_size -= 1
        font = ImageFont.truetype(FONT_PATH, font_size)
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
        otsu_thresh_val, thr_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        num_zeros = np.sum(thr_image == 0)
        num_ones = np.sum(thr_image == 255)
        thr_cnt = num_ones / (num_zeros + num_ones)

        channel_value = 0 if thr_cnt >0.5 else 255
        contrast_color.append(channel_value)

    text_color = tuple(contrast_color)

    y = y_start
    for line in wrapped_text:
        draw.text((x_start, y), line, font=font, fill=text_color)
        y += line_height + line_spacing

    img_cropped.save(output_path)

with open(PAIR_FILE, "w", newline='', encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["original_image", "synthetic_image", "text"])

    for i in tqdm(range(NUM_IMAGES), desc="Generating image pairs"):
        try:
            sample = dataset[i]
            text = sample["text"]
            original_img = sample["image"]

            orig_filename = f"{uuid4()}.png"
            synth_filename = f"{uuid4()}.png"

            orig_path = os.path.join(ORIGINAL_FOLDER, orig_filename)
            synth_path = os.path.join(SYNTHETIC_FOLDER, synth_filename)

            original_img.save(orig_path)

            bg_path = random.choice(backgrounds)
            create_synthetic_image(bg_path, text, synth_path)

            writer.writerow([orig_filename, synth_filename, text.strip()])

        except Exception as e:
            print(f"Error at index {i}: {e}")
