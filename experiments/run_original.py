# experiments/train_gmt.py
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
import argparse

def run_original(random_state_value=42):
    df = pd.read_csv("data/synthetic_data.csv")
    #df['x1_tok'] = quantile_tokenizer(df['x1'], n_bins=20)
    #df['x2_tok'] = quantile_tokenizer(df['x2'], n_bins=20)

    X = df[['x1', 'x2']]
    y = df['y']

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state_value)
    model = XGBRegressor()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"[GMT] R² score: {r2_score(y_test, y_pred):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_state_value", type=int, default=42,
                        help="Random seed for train/test split")

    args = parser.parse_args()
    run_original(random_state_value=args.random_state_value)

