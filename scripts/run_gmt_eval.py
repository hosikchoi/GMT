# -----------------------------
# scripts/run_gmt_eval.py
# -----------------------------
import argparse
import pandas as pd
import torch
from transformers import BertTokenizer, BertForMaskedLM
from gmt.downstream import run_downstream_xgboost

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, required=True)
parser.add_argument('--label_col', type=str, required=True)
args = parser.parse_args()

# Load data and model
df = pd.read_csv(args.data)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForMaskedLM.from_pretrained("./results")

# Run downstream classifier
run_downstream_xgboost(model, tokenizer, df, label_col=args.label_col)
