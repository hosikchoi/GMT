# experiments/train_gmt.py
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from utils.tokenizer import gmt_tokenizer
import argparse
import math
import os

def run_gmt(data_size=None, random_seed=None, n_bins=None, K=None, L=None):
    if data_size is None:
        raise ValueError("You must provide --data_size")

    # 파일명 생성
    data_path = f"data/synthetic_data_size{data_size}_random_seed{random_seed}.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    df = pd.read_csv(data_path)
    n = len(df)
    
    # 기본값 설정
    if K is None:
        K = 2
    # 자동 n_bins 설정        
    if n_bins is None:
        n_bins = int(math.sqrt(n))
        print(f"[Auto] n_bins_value set to √n = {n_bins}")
    if L is None:
        L = n_bins // K
        print(f"[Auto] L set to n_bins // K = {L}")
    
    # GMT 토큰화 (tuple 형태로 반환)
    df['x1_tok'] = gmt_tokenizer(df['x1'], K=K, L=L, return_format="tuple")
    df['x2_tok'] = gmt_tokenizer(df['x2'], K=K, L=L, return_format="tuple")

    # tuple -> 두 개 컬럼 분해
    df[['x1_level', 'x1_sub']] = pd.DataFrame(df['x1_tok'].tolist(), index=df.index)
    df[['x2_level', 'x2_sub']] = pd.DataFrame(df['x2_tok'].tolist(), index=df.index)
    
    ## 문자열만
    #df['x1_tok_str'] = gmt_tokenizer(df['x1'], K=3, L=5, return_format="string")
    ## 둘 다
    #tokens = gmt_tokenizer(df['x1'], K=3, L=5, return_format="dict")
    #df['x1_level'], df['x1_sub'] = pd.DataFrame(tokens["tuple"], index=df.index).T.values
    #df['x1_tok_str'] = tokens["string"]
    

    X = df[['x1_level', 'x1_sub', 'x2_level', 'x2_sub']]
    y = df['y']

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    model = XGBRegressor()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"[GMT] R² score: {r2_score(y_test, y_pred):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_size", type=int, required=True, help="Data size (used to load correct CSV file)")
    parser.add_argument("--random_seed", type=int, default=None, help="Random seed for train/test split")
    parser.add_argument("--n_bins", type=int, default=None, help="Number of bins (default: sqrt(n))")
    parser.add_argument("--K", type=int, default=None, help="Number of coarse bins (K, default: 2)")
    parser.add_argument("--L", type=int, default=None, help="Number of sub-bins (L)")

    args = parser.parse_args()
    run_gmt(data_size=args.data_size, random_seed=args.random_seed, n_bins=args.n_bins, K=args.K, L=args.L)