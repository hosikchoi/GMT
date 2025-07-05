# 환경설정
cd gmt_experiment
python -m venv venv
source venv/bin/activate   # Windows는 venv\Scripts\activate
pip install -r requirements.txt

###
gmt_experiment/
├── data/
│   └── example.csv                  # 실험용 CSV 데이터
├── gmt_tokenizer_extended.py       # GMT 계층적 토크나이저
├── wasserstein_loss.py             # Wasserstein Loss 함수
├── trainer.py                  # GMTTrainer 클래스 정의
├── run_gmt_train.py                # BERT 학습 실행 스크립트
├── visualize_embedding.py          # 토큰 임베딩 시각화
├── downstream_xgboost.py           # XGBoost 분류기
└── requirements.txt                # 필수 패키지 목록

# bash 실행 명령어
cd gmt_project
python main.py --data ./data/your_data.csv --tokenizer_method quantile --loss wasserstein --task pretrain+xgboost

# or
# bash run_all.sh
### run_all.sh
python run_gmt_train.py --data data/example.csv --numerical_cols height weight --unit_prefixes h w
python visualize_embedding.py --token_list "[level01][h3]" "[level02][h5]" "[level03][h7]" "[level04][h9]"
python downstream_xgboost.py --data data/example.csv --label_col target

# GMT + BERT MLM 학습 실행
python run_gmt_train.py --data data/example.csv --numerical_cols height weight --unit_prefixes h w


# XGBoost 다운스트림 평가 
python downstream_xgboost.py --data data/example.csv --label_col target
### --label_col: 분류할 레이블 컬럼명 (예: toxicity, disease, etc.)
### 학습된 BERT의 [CLS] embedding을 XGBoost에 입력하여 정확도 측정

