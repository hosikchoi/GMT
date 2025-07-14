# plot_gmt_embedding_tsne.py

import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

def visualize_gmt_embeddings(embedding_layer, K=5, L=10, title="GMT Token Embedding (t-SNE)"):
    """
    Parameters:
        embedding_layer: nn.Embedding (K * L, dim)
        K, L: number of levels and sub-bins
    """
    with torch.no_grad():
        embeddings = embedding_layer.weight.cpu().numpy()  # shape = (K*L, dim)

    # Compute 2D projection using t-SNE
    tsne = TSNE(n_components=2, perplexity=15, random_state=42)
    proj = tsne.fit_transform(embeddings)  # shape = (K*L, 2)

    # Assign color/label
    level = np.array([i // L for i in range(K * L)])
    sub = np.array([i % L for i in range(K * L)])

    # Plot
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(proj[:, 0], proj[:, 1], c=level, cmap='tab10', s=40, alpha=0.8)
    for i, (x, y) in enumerate(proj):
        plt.text(x, y, f"{level[i]}-{sub[i]}", fontsize=7, alpha=0.7)

    plt.title(title)
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.colorbar(scatter, label="Level (k)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

###################################################################  
from my_module import GMTNumericEmbedder
from plot_gmt_embedding_tsne import visualize_gmt_embeddings

# 예시: 학습 완료된 GMT 임베딩 모듈
gmt_embedder = GMTNumericEmbedder(K=5, L=10, emb_dim=32)
embedding_layer = gmt_embedder.embedding

# 시각화 실행
visualize_gmt_embeddings(embedding_layer, K=5, L=10)
