# models/train_fone.py
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from utils.tokenizer import fourier_embedding
import argparse
#import torch
#from number_encoders.FNE import FNE

def run_fone(data_size=None, random_seed=None, max_power=2):
    if data_size is None:
        raise ValueError("You must provide --data_size")

    # 파일명 생성
    data_path = f"data/synthetic_data_size{data_size}_random_seed{random_seed}.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)
    #fne = FNE()  # 클래스 인스턴스 생성
    #x1_embed = fne.fourier_embedding(torch.tensor(df['x1'].values))
    #x2_embed = fne.fourier_embedding(torch.tensor(df['x2'].values))
    
    x1_embed = fourier_embedding(df['x1'].values, max_power=max_power)
    x2_embed = fourier_embedding(df['x2'].values, max_power=max_power)
    
    X_embed = np.concatenate([x1_embed.T, x2_embed.T], axis=1)
    y = df['y']

    X_train, X_test, y_train, y_test = train_test_split(X_embed, y, random_state=42)
    model = XGBRegressor()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"[FoNE] R² score: {r2_score(y_test, y_pred):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()    
    parser.add_argument("--data_size", type=int, required=True, help="Data size (used to load correct CSV file)")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for train/test split")
    parser.add_argument("--max_power", type=int, default=2, help="Maximum power for Fourier embedding")
    
    args = parser.parse_args()
    run_fone(data_size=args.data_size, random_seed=args.random_seed, max_power=args.max_power)
