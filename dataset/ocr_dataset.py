# import time
# from datasets import load_dataset
# from googletrans import Translator
# from tqdm import tqdm
# import torchvision.transforms as transforms
# from PIL import Image
# import numpy as np

# # Load only 1000 samples for validation
# print("Loading validation dataset...")
# val_dataset = load_dataset("toghrultahirov/handwritten_text_ocr", split="train[:1000]")

# # Function to resize images to 224x224
# def resize_image(image):
#     if isinstance(image, dict) and 'image' in image:
#         img = image['image']
#     else:
#         img = image

#     if isinstance(img, np.ndarray):
#         img = Image.fromarray(img)

#     transform = transforms.Resize((224, 224))
#     resized_img = transform(img)
#     return resized_img

# # Resize images
# val_dataset = val_dataset.map(
#     lambda example: {"image": resize_image(example["image"])},
#     desc="Resizing validation dataset images"
# )

# # Initialize Google Translator
# translator = Translator()

# # Function to translate a single English text to Gujarati
# def translate_to_gujarati(text):
#     try:
#         return translator.translate(text, dest='gu').text
#     except Exception as e:
#         print(f"Translation failed: {text} -> {e}")
#         return ""

# # Translate and collect Gujarati text
# print("Translating texts...")
# translated_texts = []
# for item in tqdm(val_dataset, desc="Translating"):
#     gu_text = translate_to_gujarati(item['text'])
#     translated_texts.append(gu_text)
#     time.sleep(0.5)  # Delay to prevent being blocked

# # Add Gujarati translations to dataset
# val_dataset = val_dataset.add_column("text_gujarati", translated_texts)

# # Save to disk
# val_save_path = "handwritten_text_translation_validation_224x224"
# val_dataset.save_to_disk(val_save_path)
# print(f"✅ Translation validation dataset saved to: {val_save_path}")
