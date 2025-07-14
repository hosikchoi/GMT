######################################################
# 1. TrainerWithNumeric 학습 예제
######################################################
training_args = TrainingArguments(
    output_dir="./checkpoints",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    remove_unused_columns=False,  ### 중요
)
trainer = TrainerWithNumeric(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    numeric_cols=["x1", "x2"]  ### 반드시 명시
)
trainer.train()
######################################################
# 2. evaluate(), predict() 예시
######################################################
### 평가 (loss 및 metrics 리턴)
metrics = trainer.evaluate()
print(metrics)

### 예측 (logits 포함)
predictions = trainer.predict(test_dataset)
print(predictions.predictions.shape)  ### (N, vocab_size) or (N, num_labels)

### token 확장예시
from transformers import BertTokenizer
base_tok = BertTokenizer.from_pretrained("bert-base-uncased")
gmt_tokens = [f"[level{k}][l{l}]" for k in range(5) for l in range(10)]
base_tok.add_tokens(gmt_tokens, special_tokens=False)

### resize
model.resize_token_embeddings(len(base_tok))

######################################################
# 3. tokenizer와 model 저장 (save_pretrained)
######################################################
### tokenizer 저장
tokenizer.save_pretrained("checkpoints/tokenizer/")
### 모델 저장 (사전학습 or fine-tuning 후)
model.save_pretrained("checkpoints/model/")
- checkpoints/tokenizer/vocab.txt
- checkpoints/tokenizer/tokenizer_config.json
- checkpoints/model/pytorch_model.bin
- checkpoints/model/config.json

######################################################
# 4. 저장된 tokenizer/model 다시 불러오기
######################################################
from transformers import BertTokenizer, BertConfig
from my_module import BertForPreTrainingWithNumeric

tokenizer = BertTokenizer.from_pretrained("checkpoints/tokenizer/")
config = BertConfig.from_pretrained("checkpoints/model/")

# GMT 임베딩 정보 수동으로 다시 지정
gmt_embedder = GMTNumericEmbedder(K=5, L=10, emb_dim=config.hidden_size, return_format="string")
numeric_cols = ["x1", "x2"]

model = BertForPreTrainingWithNumeric(config, gmt_embedder, numeric_cols)
model.load_state_dict(torch.load("checkpoints/model/pytorch_model.bin"), strict=False)

######################################################
from transformers import BertTokenizer, BertConfig
from my_module import BertForPreTrainingWithNumeric

tokenizer = BertTokenizer.from_pretrained("checkpoints/tokenizer/")
config = BertConfig.from_pretrained("checkpoints/model/")

### GMT 임베딩 정보 수동으로 다시 지정
gmt_embedder = GMTNumericEmbedder(K=5, L=10, emb_dim=config.hidden_size, return_format="string")
numeric_cols = ["x1", "x2"]

model = BertForPreTrainingWithNumeric(config, gmt_embedder, numeric_cols)
model.load_state_dict(torch.load("checkpoints/model/pytorch_model.bin"), strict=False)

######################################################
# 3. downstream classification 예제
######################################################
from my_module import BertForSequenceClassificationWithNumeric

# config: 기존 BERT config
config.num_labels = 2  # binary classification
model_cls = BertForSequenceClassificationWithNumeric(config, gmt_embedder, numeric_cols)

# TrainerWithNumeric은 그대로 사용
trainer_cls = TrainerWithNumeric(
    model=model_cls,
    args=training_args,
    train_dataset=clf_train_dataset,
    eval_dataset=clf_eval_dataset,
    tokenizer=tokenizer,
    data_collator=default_data_collator,  # classification에서는 masking X
    numeric_cols=numeric_cols
)

trainer_cls.train()
metrics = trainer_cls.evaluate()
print(metrics)





















