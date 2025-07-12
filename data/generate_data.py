# data/generate_data.py
import numpy as np
import pandas as pd
import torch

def generate_linear_data(n=1000, noise_std=0.1, seed=42):
    np.random.seed(seed)
    x1 = np.random.uniform(0, 1, n)
    x2 = np.random.uniform(0, 1, n)
    y = 2 * x1 - 3 * x2 + np.random.normal(0, noise_std, n)
    
    df = pd.DataFrame({'x1': x1, 'x2': x2, 'y': y})
    return df

if __name__ == "__main__":
    df = generate_linear_data()
    df.to_csv("data/synthetic_data.csv", index=False)
    tensor = torch.tensor(df.values, dtype=torch.float32)
    torch.save(tensor, "data/synthetic_tensor.pt")

