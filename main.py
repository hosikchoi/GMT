# main.py
import argparse
import pandas as pd
from gmt_tokenizer import GeneralMeasureTokenizer, tokenize_numerical_column
from transformers import BertTokenizer, BertForMaskedLM
from trainer import GMTTrainer, NumericDataset
from transformers import TrainingArguments
import torch
from sklearn.model_selection import train_test_split
from downstream_xgboost import run_downstream_xgboost

def build_dataset(texts, tokenizer):
    encodings = tokenizer(texts, padding="max_length", truncation=True, return_tensors="pt")
    labels = encodings["input_ids"].clone()
    return NumericDataset(encodings, labels)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--tokenizer_method', type=str, default="quantile", choices=["quantile", "gauss_rank"])
    parser.add_argument('--loss', type=str, default="wasserstein", choices=["wasserstein", "mse"])
    parser.add_argument('--task', type=str, default="pretrain+xgboost")
    args = parser.parse_args()

    # 1. Load CSV and preprocess
    df = pd.read_csv(args.data)
    num_col = [col for col in df.columns if df[col].dtype in ['float64', 'int64']][0]
    labels = df["target"] if "target" in df.columns else None

    # 2. Tokenize numerical values
    gmt = GeneralMeasureTokenizer(method=args.tokenizer_method, num_bins=10)
    tokens = tokenize_numerical_column(df[num_col].values, gmt, unit_prefix="l")
    df["text"] = tokens

    # 3. Train/Test split
    train_texts, val_texts, train_labels, val_labels = train_test_split(df["text"].tolist(), df["text"].tolist(), test_size=0.2)

    # 4. HuggingFace tokenizer and model
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForMaskedLM.from_pretrained("bert-base-uncased")

    train_dataset = build_dataset(train_texts, tokenizer)
    val_dataset = build_dataset(val_texts, tokenizer)

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        evaluation_strategy="epoch",
        logging_dir="./results/logs",
        save_strategy="epoch"
    )

    # 5. Trainer
    trainer = GMTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    trainer.train()

    # 6. Optional downstream classification
    if "xgboost" in args.task and labels is not None:
        df["target"] = labels
        run_downstream_xgboost(model, tokenizer, df, label_col="target")

if __name__ == "__main__":
    main()
