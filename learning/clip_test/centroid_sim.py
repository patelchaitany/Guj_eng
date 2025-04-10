import torch
import clip
from PIL import Image
import pandas as pd
import os
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

df = pd.read_csv("pairs.csv")

original_embeddings = []
synthetic_embeddings = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    try:
        orig_path = os.path.join("original_images", row["original_image"])
        synt_path = os.path.join("synthetic_images", row["synthetic_image"])

        orig_img = preprocess(Image.open(orig_path)).unsqueeze(0).to(device)
        synt_img = preprocess(Image.open(synt_path)).unsqueeze(0).to(device)

        with torch.no_grad():
            orig_emb = model.encode_image(orig_img)
            synt_emb = model.encode_image(synt_img)

        orig_emb /= orig_emb.norm(dim=-1, keepdim=True)
        synt_emb /= synt_emb.norm(dim=-1, keepdim=True)

        original_embeddings.append(orig_emb)
        synthetic_embeddings.append(synt_emb)

    except Exception as e:
        print(f"Skipping pair due to error: {e}")

orig_tensor = torch.cat(original_embeddings, dim=0)
synt_tensor = torch.cat(synthetic_embeddings, dim=0)

centroid_orig = orig_tensor.mean(dim=0)
centroid_synt = synt_tensor.mean(dim=0)

centroid_orig /= centroid_orig.norm()
centroid_synt /= centroid_synt.norm()

centroid_similarity = (centroid_orig @ centroid_synt).item()

print(f"\nCosine similarity between centroid of original and synthetic images: {centroid_similarity:.4f}")
