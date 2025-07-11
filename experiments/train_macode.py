# experiments/train_macode.py
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from utils.tokenizer import histogram_tokenizer

def run_macode():
    df = pd.read_csv("data/synthetic_data.csv")
    df['x1_tok'] = histogram_tokenizer(df['x1'], n_bins=50)
    df['x2_tok'] = histogram_tokenizer(df['x2'], n_bins=50)

    X = df[['x1_tok', 'x2_tok']]
    y = df['y']

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    model = XGBRegressor()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"[MaCoDE] R² score: {r2_score(y_test, y_pred):.4f}")

if __name__ == "__main__":
    run_macode()
