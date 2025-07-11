# downstream_macode.py
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from models.pretrain_macode import MaCoDEModel, MaCoDETokenizer

def run_downstream_macode():
    df = pd.read_csv("data/synthetic_data.csv")
    tokenizer = MaCoDETokenizer(n_bins=50)
    x1 = tokenizer.fit_transform(df['x1'].values)
    x2 = tokenizer.fit_transform(df['x2'].values)
    tokens = torch.tensor(np.stack([x1, x2], axis=1), dtype=torch.long)

    model = MaCoDEModel()
    model.load_state_dict(torch.load("models/macode_bert.pt"))
    model.eval()

    with torch.no_grad():
        x_emb = model.emb(tokens)  # (N, T, H)
        x_vec = x_emb.mean(dim=1).numpy()

    y = df['y'].values
    X_train, X_test, y_train, y_test = train_test_split(x_vec, y, test_size=0.2, random_state=42)
    reg = Ridge().fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    print(f"[MaCoDE Downstream] R² score: {r2_score(y_test, y_pred):.4f}")

if __name__ == "__main__":
    run_downstream_macode()
