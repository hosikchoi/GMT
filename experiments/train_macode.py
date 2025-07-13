# experiments/train_macode.py
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from utils.tokenizer import quantile_tokenizer
import argparse
import math
import os

def run_macode(data_size=None, random_seed=None, n_bins=None):
    if data_size is None:
        raise ValueError("You must provide --data_size")

    # 파일명 생성
    data_path = f"data/synthetic_data_size{data_size}_random_seed{random_seed}.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    df = pd.read_csv(data_path)
    n = len(df)

    # 자동 n_bins 설정
    if n_bins is None:
        n_bins = int(math.sqrt(n))
        print(f"[Auto] n_bins_value set to √n = {n_bins}")

    #df['x1_tok'] = histogram_tokenizer(df['x1'], n_bins=n_bins)
    #df['x2_tok'] = histogram_tokenizer(df['x2'], n_bins=n_bins)
    df['x1_tok'] = quantile_tokenizer(df['x1'], n_bins=n_bins)
    df['x2_tok'] = quantile_tokenizer(df['x2'], n_bins=n_bins)

    X = df[['x1_tok', 'x2_tok']]
    y = df['y']

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    model = XGBRegressor()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"[MaCoDE] R² score: {r2_score(y_test, y_pred):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_size", type=int, required=True, help="Data size (used to load correct CSV file)")
    parser.add_argument("--random_seed", type=int, default=None, help="Random seed for train/test split")
    parser.add_argument("--n_bins", type=int, default=None, help="Number of bins (default: sqrt(n))")

    args = parser.parse_args()
    run_macode(data_size=args.data_size, random_seed=args.random_seed, n_bins=args.n_bins)
