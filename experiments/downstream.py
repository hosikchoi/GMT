# experiments/downstream.py
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from utils.gmt_tokenizer import GeneralMeasureTokenizer
from models.pretrain_gmt import NumericMLM

def extract_cls_embedding(model, token_tensor):
    with torch.no_grad():
        x_emb = model.emb(token_tensor)
        output = model.bert(inputs_embeds=x_emb).last_hidden_state
        return output[:, 0, :].numpy()  # [CLS] token

def run_downstream():
    df = pd.read_csv("data/synthetic_data.csv")
    y = df['y'].values

    tokenizer = GeneralMeasureTokenizer()
    x1 = tokenizer.fit_transform(df['x1'].values)
    x2 = tokenizer.fit_transform(df['x2'].values)
    x_tok = torch.stack([x1[:, 0] * 10 + x1[:, 1], x2[:, 0] * 10 + x2[:, 1]], dim=1)  # vocab index

    model = NumericMLM()
    model.load_state_dict(torch.load("models/pretrained_bert.pt"))
    model.eval()

    embedding = extract_cls_embedding(model, x_tok)

    X_train, X_test, y_train, y_test = train_test_split(embedding, y, test_size=0.2, random_state=0)
    reg = Ridge()
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    print(f"[Downstream] R² score: {r2_score(y_test, y_pred):.4f}")

if __name__ == "__main__":
    run_downstream()
