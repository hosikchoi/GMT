# BERT-style Pretraining with Wasserstein Loss
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertConfig
from utils.gmt_tokenizer import GeneralMeasureTokenizer
from utils.loss import wasserstein_loss

class NumericDataset(Dataset):
    def __init__(self, X_tok):
        self.X = X_tok
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, idx): return self.X[idx]

class NumericMLM(nn.Module):
    def __init__(self, vocab_size=100, hidden_dim=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, hidden_dim)
        config = BertConfig(hidden_size=hidden_dim, num_hidden_layers=2, num_attention_heads=4, intermediate_size=256)
        self.bert = BertModel(config)
        self.cls_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):  # x: (B, T)
        x_emb = self.emb(x)
        output = self.bert(inputs_embeds=x_emb).last_hidden_state
        logits = self.cls_head(output)
        return logits  # (B, T, V)

def pretrain():
    df = torch.load("data/synthetic_tensor.pt")  # tensor of shape (N, 2)
    tokenizer = GeneralMeasureTokenizer()
    token_ids = tokenizer.fit_transform(df)

    dataset = NumericDataset(token_ids)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = NumericMLM()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for epoch in range(10):
        model.train()
        for batch in loader:
            mask = torch.rand(batch.shape) < 0.15
            input_ids = batch.clone()
            input_ids[mask] = 99  # [MASK] index

            logits = model(input_ids)
            loss = wasserstein_loss(logits, batch, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"[epoch {epoch}] loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "models/pretrained_bert.pt")

if __name__ == "__main__":
    pretrain()
