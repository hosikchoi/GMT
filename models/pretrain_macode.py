import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from models.loss import wasserstein_loss

class MaCoDETokenizer:
    def __init__(self, n_bins=50):
        self.n_bins = n_bins
        self.bins = None

    def fit(self, x):
        x = np.array(x)
        x_log = np.log1p(x)
        self.xmin, self.xmax = x_log.min(), x_log.max()
        self.bins = np.linspace(0, 1, self.n_bins + 1)
        return self

    def transform(self, x):
        x_log = np.log1p(x)
        x_norm = (x_log - self.xmin) / (self.xmax - self.xmin)
        return np.digitize(x_norm, self.bins) - 1  # 0-based index

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

class MaCoDEDataset(Dataset):
    def __init__(self, token_ids):
        self.tokens = token_ids  # (N, T)

    def __len__(self):
        return self.tokens.shape[0]

    def __getitem__(self, idx):
        return self.tokens[idx]

class MaCoDEModel(nn.Module):
    def __init__(self, vocab_size=50, hidden_dim=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size + 1, hidden_dim)  # +1 for [MASK]=0
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4),
            num_layers=2
        )
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):  # x: (B, T)
        x_emb = self.emb(x)         # (B, T, H)
        x_tr = self.transformer(x_emb.permute(1, 0, 2))  # (T, B, H)
        logits = self.head(x_tr.permute(1, 0, 2))  # (B, T, V)
        return logits

def pretrain_macode():
    df = pd.read_csv("data/synthetic_data.csv")
    tokenizer = MaCoDETokenizer(n_bins=50)
    x1 = tokenizer.fit_transform(df['x1'].values)
    x2 = tokenizer.fit_transform(df['x2'].values)
    tokens = np.stack([x1, x2], axis=1)
    tokens_tensor = torch.tensor(tokens, dtype=torch.long)

    dataset = MaCoDEDataset(tokens_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = MaCoDEModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    mask_token_id = 0  # reserved for [MASK]

    for epoch in range(10):
        model.train()
        for batch in loader:
            input_ids = batch.clone()
            mask = torch.rand(input_ids.shape) < 0.15
            input_ids[mask] = mask_token_id

            logits = model(input_ids)  # (B, T, V)
            loss = wasserstein_loss(logits, batch, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"[MaCoDE-epoch {epoch}] loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "models/macode_bert.pt")

if __name__ == "__main__":
    pretrain_macode()
