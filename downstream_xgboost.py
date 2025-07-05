# downstream_xgboost.py
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np

def run_downstream_xgboost(model, tokenizer, df, label_col="target"):
    model.eval()
    texts = df["text"].tolist()
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # [CLS] token

    X_train, X_test, y_train, y_test = train_test_split(cls_embeddings, df[label_col], test_size=0.2, random_state=42)

    clf = GradientBoostingClassifier()
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Downstream XGBoost Accuracy: {acc:.4f}")
