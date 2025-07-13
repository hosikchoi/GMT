# models/train_fone.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from utils.tokenizer import fourier_embedding
import argparse
#import torch
#from number_encoders.FNE import FNE

def run_fone(max_power_value=2, random_state_value=42):
    df = pd.read_csv("data/synthetic_data.csv")
    
    #fne = FNE()  # 클래스 인스턴스 생성
    #x1_embed = fne.fourier_embedding(torch.tensor(df['x1'].values))
    #x2_embed = fne.fourier_embedding(torch.tensor(df['x2'].values))
    
    x1_embed = fourier_embedding(df['x1'].values, max_power=max_power_value)
    x2_embed = fourier_embedding(df['x2'].values, max_power=max_power_value)
    
    X_embed = np.concatenate([x1_embed.T, x2_embed.T], axis=1)
    y = df['y']
    print(X_embed.shape)
    X_train, X_test, y_train, y_test = train_test_split(X_embed, y, 
                                                        random_state=random_state_value)
    model = XGBRegressor()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"[FoNE] R² score: {r2_score(y_test, y_pred):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_power_value", type=int, default=2,
                        help="Maximum power for Fourier embedding")
    parser.add_argument("--random_state_value", type=int, default=42,
                        help="Random seed for train/test split")

    args = parser.parse_args()
    run_fone(max_power_value=args.max_power_value, random_state_value=args.random_state_value)
