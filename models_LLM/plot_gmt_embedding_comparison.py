import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np


def extract_embedding_numpy(embedding_layer):
    return embedding_layer.weight.detach().cpu().numpy()

def compute_tsne_projection(embeddings, n_components=2, perplexity=15, seed=42):
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=seed)
    return tsne.fit_transform(embeddings)

def plot_embedding_comparison(
    before, after, K, L, title="GMT Embedding: Before vs. After Training"
):
    """
    Parameters:
        before: np.ndarray of shape (K*L, dim)
        after: np.ndarray of shape (K*L, dim)
    """
    proj_before = compute_tsne_projection(before)
    proj_after = compute_tsne_projection(after)

    level = np.array([i // L for i in range(K * L)])
    sub = np.array([i % L for i in range(K * L)])

    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(K * L):
        x0, y0 = proj_before[i]
        x1, y1 = proj_after[i]
        ax.arrow(
            x0, y0, x1 - x0, y1 - y0,
            head_width=0.4, alpha=0.5,
            color=plt.cm.tab10(level[i] % 10)
        )
        ax.text(x1, y1, f"{level[i]}-{sub[i]}", fontsize=7, alpha=0.7)

    ax.set_title(title)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#############################################################################
from my_module import GMTNumericEmbedder
from plot_gmt_embedding_comparison import extract_embedding_numpy, plot_embedding_comparison
# 모델 초기화 및 학습 전후 임베딩 추출
K, L = 5, 10
embedder = GMTNumericEmbedder(K=K, L=L, emb_dim=32)
embedding_layer = embedder.embedding

# 학습 전 저장
emb_before = extract_embedding_numpy(embedding_layer)
# 학습 예시 (embedding_layer.weight가 업데이트된 상태)
# trainer.train() 수행
# 학습 후 저장
emb_after = extract_embedding_numpy(embedding_layer)
# 시각화
plot_embedding_comparison(emb_before, emb_after, K=K, L=L)
                  


