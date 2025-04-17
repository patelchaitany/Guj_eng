import torch
import os
from datasets import load_from_disk
from torch.nn import functional as F
from torchvision import transforms
from model.transformer import Transformer
from model.config import Config
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm
from tokenizer.tokenizer import Tokenizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

special_token = ["<en|gu>", "<gu|en>","<en>", "<gu>"]
tokenizer_path = "data/tokenizer.model"
tokenizer = Tokenizer(special_token)
tokenizer.load(tokenizer_path)

vocab_size = tokenizer.get_vocab_size()
config = Config(
    vocab_size = vocab_size
)

model = Transformer(config).to(device)

model.load_state_dict(torch.load("checkpoints/transformer_epoch_2649.pt", map_location=device)['model_state_dict'])
model.eval()
print("Model and weights loaded.")

dataset_path = "test_ting_data"
test_dataset = load_from_disk(dataset_path)
print(f"Loaded {len(test_dataset)} samples from test dataset.")

preprocess = transforms.Compose([
    transforms.ToTensor(),
])

def predict_image(model, image_tensor, tokenizer, max_len=32):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)
        x = torch.tensor([[tokenizer.get_token_id("bos"),tokenizer.get_token_id("<gu>")]], device=device)

        for _ in range(max_len):
            output = model(x, image_tensor, is_image=True)
            next_token_logits = output[:, -1, :]
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)

            if next_token.item() == tokenizer.get_token_id("eos"):
                break
            x = torch.cat([x, next_token], dim=1)

        tokens = x.squeeze().tolist()
        return tokenizer.decode(tokens)

bleu1_scores = []
bleu2_scores = []
bleu3_scores = []
bleu4_scores = []

smoothie = SmoothingFunction().method4

for i, item in enumerate(tqdm(test_dataset, desc="Evaluating BLEU")):
    image = item["image"]
    reference = item["text_gujarati"]

    image_tensor = preprocess(image)
    predicted_text = predict_image(model, image_tensor, tokenizer)

    ref_tokens = reference.split()
    pred_tokens = predicted_text.split()

    bleu1 = sentence_bleu([ref_tokens], pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothie)
    bleu2 = sentence_bleu([ref_tokens], pred_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
    bleu3 = sentence_bleu([ref_tokens], pred_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothie)
    bleu4 = sentence_bleu([ref_tokens], pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)

    bleu1_scores.append(bleu1)
    bleu2_scores.append(bleu2)
    bleu3_scores.append(bleu3)
    bleu4_scores.append(bleu4)

    if i % 100 == 0:
        print(f"\nSample {i}:")
        print(f"Reference: {reference}")
        print(f"Predicted: {predicted_text}")
        print(f"BLEU-1: {bleu1:.4f}, BLEU-2: {bleu2:.4f}, BLEU-3: {bleu3:.4f}, BLEU-4: {bleu4:.4f}")

print("\n Final BLEU Scores:")
print(f"BLEU-1: {sum(bleu1_scores)/len(bleu1_scores):.4f}")
print(f"BLEU-2: {sum(bleu2_scores)/len(bleu2_scores):.4f}")
print(f"BLEU-3: {sum(bleu3_scores)/len(bleu3_scores):.4f}")
print(f"BLEU-4: {sum(bleu4_scores)/len(bleu4_scores):.4f}")


from PIL import Image
import uuid

output_folder = "generated_samples"
os.makedirs(output_folder, exist_ok=True)

num_samples_to_save = 10
print("\nSaving sample predictions:\n")

for idx, item in enumerate(test_dataset.select(range(num_samples_to_save))):
    image = item["image"]
    image_tensor = preprocess(image)
    predicted_text = predict_image(model, image_tensor, tokenizer)

    img_name = f"sample_{idx}_{uuid.uuid4().hex[:6]}.png"
    img_path = os.path.join(output_folder, img_name)

    image.save(img_path)

    print(f"{img_name} -> {predicted_text}")
