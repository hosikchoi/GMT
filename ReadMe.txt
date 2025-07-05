# bash 실행 명령어
cd gmt_project
python main.py --data ./data/your_data.csv --tokenizer_method quantile --loss wasserstein --task pretrain+xgboost
