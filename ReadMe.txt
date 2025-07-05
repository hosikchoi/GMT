# 전체 실행 순서 개요
1. 환경 설정
  cd gmt_experiment
  python -m venv venv
  source venv/bin/activate     # (Windows: venv\Scripts\activate)
  pip install -r requirement.txt

2. 데이터 준비 및 토크나이징 (GMT 기반)
  python main.py \
    --data ./data/data.csv \
    --tokenizer_method quantile \
    --loss wasserstein \
    --task pretrain+xgboost
### --tokenizer_method: quantile 또는 gauss_rank 방식 선택
### --loss: wasserstein 또는 mse 선택
### --task: pretrain, xgboost, pretrain+xgboost 등
### gmt_tokenizer.py의 GeneralMeasureTokenizer를 이용하여 수치형 컬럼을 [level-k][u-l] 구조의 토큰으로 변

3. BERT 기반 MLM 사전학습
### trainer.py 내 GMTTrainer에서 ntl_wasserstein_loss 또는 ntl_mse_loss 기반으로 학습을 수행

4. 임베딩 시각화 (optional)
  python visualize_embedding.py
### plot_token_embeddings(model, tokenizer, token_list)를 이용해 tsne로 임베딩 간 L2 거리 시각화

5. Downstream XGBoost 분류기 평가
  python downstream_xgboost.py --data ./data/data.csv --label_col target

###
gmt_experiment/
├── data/
│   └── example.csv                 # 실험용 CSV 데이터
├── results/
│   └── logs/, chekpoints           # 결과 저장 폴더
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

###################################################################
# 1. 환경설정
cd gmt_experiment
python -m venv venv
source venv/bin/activate   # Windows는 venv\Scripts\activate
pip install -r requirements.txt

# 2. GMT + BERT MLM 학습 실행
python run_gmt_train.py --data data/example.csv --numerical_cols height weight --unit_prefixes h w
--data: 입력 데이터 CSV 경로
--numerical_cols: 수치형 컬럼들
--unit_prefixes: 각 변수에 대응되는 unit prefix (h for height, w for weight)

# 3. 임베딩 시각화 실행
python visualize_embedding.py --token_list "[level01][h3]" "[level02][h5]" "[level03][h7]" "[level04][h9]"

# XGBoost 다운스트림 평가 
python downstream_xgboost.py --data data/example.csv --label_col target
### --label_col: 분류할 레이블 컬럼명 (예: toxicity, disease, etc.)
### 학습된 BERT의 [CLS] embedding을 XGBoost에 입력하여 정확도 측정

