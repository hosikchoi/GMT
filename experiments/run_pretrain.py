import os

print("Step 1: Pretraining BERT with GMT")
os.system("models/pretrain_gmt.py")

print("Step 2: Evaluate on downstream task")
os.system("models/downstream.py")
