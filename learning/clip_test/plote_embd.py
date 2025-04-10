import torch
import clip
from PIL import Image
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

        original_embeddings.append(orig_emb.cpu().numpy())
        synthetic_embeddings.append(synt_emb.cpu().numpy())

    except Exception as e:
        print(f"Skipping pair due to error: {e}")

original_embeddings = torch.tensor(original_embeddings).squeeze(1).numpy()
synthetic_embeddings = torch.tensor(synthetic_embeddings).squeeze(1).numpy()

all_embeddings = np.concatenate([original_embeddings, synthetic_embeddings], axis=0)
labels = ['original'] * len(original_embeddings) + ['synthetic'] * len(synthetic_embeddings)

tsne = TSNE(n_components=3, random_state=42, perplexity=30)
embeddings_3d = tsne.fit_transform(all_embeddings)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

colors = ['red' if label == 'original' else 'blue' for label in labels]

ax.scatter(embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2], c=colors, alpha=0.6)
ax.set_title("3D t-SNE of Original vs Synthetic Image Embeddings")
ax.legend(['Original (red)', 'Synthetic (blue)'])
plt.tight_layout()
plt.show()
