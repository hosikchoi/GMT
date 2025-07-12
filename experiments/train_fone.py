# models/train_fone.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

#import torch
#from number_encoders.FNE import FNE
from utils.tokenizer import fourier_embedding

def run_fone():
    df = pd.read_csv("data/synthetic_data.csv")
    
    #fne = FNE()  # 클래스 인스턴스 생성
    #x1_embed = fne.fourier_embedding(torch.tensor(df['x1'].values))
    #x2_embed = fne.fourier_embedding(torch.tensor(df['x2'].values))
    
    x1_embed = fourier_embedding(df['x1'].values, max_power=2)
    x2_embed = fourier_embedding(df['x2'].values, max_power=2)

    X_embed = np.concatenate([x1_embed, x2_embed], axis=1)
    y = df['y']

    X_train, X_test, y_train, y_test = train_test_split(X_embed, y, random_state=42)
    model = XGBRegressor()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"[FoNE] R² score: {r2_score(y_test, y_pred):.4f}")

if __name__ == "__main__":
    run_fone()

