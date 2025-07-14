import torch
import torch.nn as nn
import numpy as np
import pandas as pd

class GMTNumericEmbedder(nn.Module):
    def __init__(self, K=2, L=10, emb_dim=32, return_format="string"):
        super().__init__()
        self.K = K
        self.L = L
        self.emb_dim = emb_dim
        self.return_format = return_format

        self.vocab = self.build_vocab(K, L, format=return_format)
        self.tok2id = {tok: idx for idx, tok in enumerate(self.vocab)}
        self.embedding = nn.Embedding(len(self.vocab), emb_dim)

    def build_vocab(self, K, L, format="string"):
        if format == "string":
            return [f"[level{k}][l{l}]" for k in range(K) for l in range(L)]
        elif format == "tuple":
            return [(k, l) for k in range(K) for l in range(L)]

    def gmt_tokenizer(self, values):
        values = np.asarray(values)
        x_min, x_max = np.min(values), np.max(values)
        x_prime = (np.log1p(values) - np.log1p(x_min)) / (np.log1p(x_max) - np.log1p(x_min))
        x_prime = np.clip(x_prime, 0, 1 - 1e-8)

        total_bins = self.K * self.L
        bin_idx = (x_prime * total_bins).astype(int)
        level_k = bin_idx // self.L
        sub_l = bin_idx % self.L

        if self.return_format == "tuple":
            return list(zip(level_k, sub_l))
        else:  # "string"
            return [f"[level{l}][l{s}]" for l, s in zip(level_k, sub_l)]

    def tokenize_dataframe(self, df: pd.DataFrame, numeric_cols: list):
        tok_df = {}
        for col in numeric_cols:
            tok_df[col] = self.gmt_tokenizer(df[col])
        return pd.DataFrame(tok_df)

    def forward(self, df: pd.DataFrame, numeric_cols: list):
        # 1. tokenize
        df_tok = self.tokenize_dataframe(df, numeric_cols)

        # 2. token → id
        df_tok_ids = df_tok.applymap(lambda tok: self.tok2id[tok])
        token_tensor = torch.tensor(df_tok_ids.values, dtype=torch.long)  # (batch, n_feat)

        # 3. embedding lookup
        emb = self.embedding(token_tensor)  # (batch, n_feat, emb_dim)
        return emb

## 데이터프레임
#df = pd.DataFrame({
#    'x1': np.random.rand(1000) * 50,
#    'x2': np.random.rand(1000) * 100,
#    'y': np.random.rand(1000)
#})

## 수치형 feature 선택
#numeric_cols = ['x1', 'x2']
## GMT embedder 초기화
#gmt_embedder = GMTNumericEmbedder(K=2, L=10, emb_dim=32, return_format="string")
#또는 
#embedder = GMTNumericEmbedder(K=2, L=10, emb_dim=32)
#numeric_emb = embedder(df, ['x1', 'x2'])  # → (batch, 2, 32)








## embedding vector 추출
#numeric_embeds = gmt_embedder(df, numeric_cols)  # shape = (1000, 2, 32)

