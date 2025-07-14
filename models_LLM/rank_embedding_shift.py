# 이동 거리 분
import torch
import numpy as np
import matplotlib.pyplot as plt

def rank_embedding_shifts(emb_before, emb_after, K, L, top_k=10, print_result=True):
    """
    emb_before, emb_after: np.ndarray of shape (K*L, dim)
    """
    diffs = emb_after - emb_before  # (K*L, dim)
    distances = np.linalg.norm(diffs, axis=1)  # (K*L,)

    sorted_indices = np.argsort(-distances)  # 내림차순
    top_indices = sorted_indices[:top_k]

    results = []
    for idx in top_indices:
        k, l = idx // L, idx % L
        results.append((k, l, distances[idx]))

    if print_result:
        print(f"🔝 Top {top_k} shifting GMT tokens:")
        for i, (k, l, d) in enumerate(results):
            print(f"{i+1:>2}. [level{k}][l{l}] → Δdistance: {d:.4f}")

    return results, distances

###############################################################################
from plot_gmt_embedding_comparison import extract_embedding_numpy
from rank_embedding_shift import rank_embedding_shifts

emb_before = extract_embedding_numpy(embedding_layer)
# 학습 수행 후...
emb_after = extract_embedding_numpy(embedding_layer)

top_tokens, all_distances = rank_embedding_shifts(emb_before, emb_after, K=5, L=10, top_k=10)

#################################################################################
def plot_shifted_tokens_bar(top_tokens, title="Top Token Embedding Shifts"):
    labels = [f"[{k}-{l}]" for k, l, _ in top_tokens]
    shifts = [d for _, _, d in top_tokens]

    plt.figure(figsize=(8, 4))
    plt.barh(labels[::-1], shifts[::-1], color="tomato")
    plt.xlabel("L2 Shift Distance")
    plt.title(title)
    plt.tight_layout()
    plt.show()
###################################################################################
top_tokens, _ = rank_embedding_shifts(emb_before, emb_after, K=5, L=10, top_k=10)
plot_shifted_tokens_bar(top_tokens)

