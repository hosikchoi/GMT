# visualize_embedding.py
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import torch

def plot_token_embeddings(model, tokenizer, token_list):
    model.eval()
    token_ids = tokenizer.convert_tokens_to_ids(token_list)
    with torch.no_grad():
        embeddings = model.bert.embeddings.word_embeddings(torch.tensor(token_ids))

    vecs = embeddings.cpu().numpy()
    dist_matrix = np.linalg.norm(vecs[:, None] - vecs[None, :], axis=-1)

    tsne = TSNE(n_components=2)
    vecs_2d = tsne.fit_transform(vecs)

    plt.figure(figsize=(8, 6))
    for i, token in enumerate(token_list):
        plt.scatter(vecs_2d[i, 0], vecs_2d[i, 1])
        plt.text(vecs_2d[i, 0]+0.01, vecs_2d[i, 1]+0.01, token)
    plt.title("Token Embedding Visualization (L2 Distance)")
    plt.show()

    return dist_matrix

"""
token_list = [f"[level{l:02d}][l{k}]" for l in range(5) for k in range(4)]
dist_matrix = plot_token_embeddings(model, tokenizer, token_list)
"""
