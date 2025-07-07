#!/bin/bash
python scripts/run_gmt_train.py --data data/data.csv --numerical_cols height weight --unit_prefixes h w
python visualize/visualize_embedding.py --token_list "[level01][h3]" "[level02][h5]" "[level03][h7]" "[level04][h9]"
python gmt/downstream_xgboost.py --data data/data.csv --label_col target
