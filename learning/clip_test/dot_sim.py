import torch
import clip
from PIL import Image
import pandas as pd
import os
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

df = pd.read_csv("pairs.csv")

similarities = []

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

        sim = (orig_emb @ synt_emb.T).item()

    except Exception as e:
        print(f"Error processing {row['original_image']} & {row['synthetic_image']}: {e}")
        sim = None 

    similarities.append(sim)

df["similarity"] = similarities
df.to_csv("pairs_with_similarity.csv", index=False)

valid_sims = [s for s in similarities if s is not None]
print(f"\nProcessed {len(valid_sims)} image pairs.")
print(f"Average similarity: {sum(valid_sims)/len(valid_sims):.4f}")
