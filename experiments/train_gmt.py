# experiments/train_gmt.py
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from utils.tokenizer import quantile_tokenizer
import argparse

def run_gmt(n_bins_value=20, random_state_value=42):
    df = pd.read_csv("data/synthetic_data.csv")
    df['x1_tok'] = quantile_tokenizer(df['x1'], n_bins=n_bins_value)
    df['x2_tok'] = quantile_tokenizer(df['x2'], n_bins=n_bins_value)

    X = df[['x1_tok', 'x2_tok']]
    y = df['y']

    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        random_state=random_state_value)
    model = XGBRegressor()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"[Original] R² score: {r2_score(y_test, y_pred):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_bins_value", type=int, default=20,
                        help="Number of bins")
    parser.add_argument("--random_state_value", type=int, default=42,
                        help="Random seed for train/test split")

    args = parser.parse_args()
    run_gmt(n_bins_value=args.n_bins_value, random_state_value=args.random_state_value)

