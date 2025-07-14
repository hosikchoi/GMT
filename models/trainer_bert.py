# 1. 데이터준비 (text+수치)
from transformers import BertTokenizer, BertConfig
from transformers import TrainingArguments, Trainer
from transformers.data.data_collator import default_data_collator

from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class TextWithNumericDataset(Dataset):
    def __init__(self, texts, numerics, labels=None):
        self.texts = texts
        self.numerics = numerics
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        example = {
            "text": self.texts[idx],
        }
        for key in self.numerics.columns:
            example[key] = self.numerics.iloc[idx][key]
        if self.labels is not None:
            example["label"] = self.labels[idx]
        return example

# 2. tokenizer, model, embedder 생성
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
config = BertConfig()

### 수치형 embedder (공통 vocabulary 사용)
from my_module import GMTNumericEmbedder  # or 직접 정의
gmt_embedder = GMTNumericEmbedder(K=2, L=10, emb_dim=config.hidden_size, return_format="string")
numeric_cols = ["x1", "x2"]

### 모델 선택
from my_module import BertForPreTrainingWithNumeric  # or BertForSequenceClassificationWithNumeric
model = BertForPreTrainingWithNumeric(config, gmt_embedder, numeric_cols)

# 3. 데이터 준비
texts = ["this is a test", "another sample", "more data"] * 100
numerics = pd.DataFrame({
    "x1": np.random.rand(len(texts)) * 100,
    "x2": np.random.rand(len(texts)) * 50,
})
labels = None  # MLM에서는 label이 input_ids로부터 생성됨
dataset = TextWithNumericDataset(texts, numerics, labels)

# 3. DataCollator 지정 (MLM용)
from my_module import DataCollatorForNumericMLM
data_collator = DataCollatorForNumericMLM(
    tokenizer=tokenizer,
    gmt_embedder=gmt_embedder,
    numeric_cols=numeric_cols,
    mlm_prob=0.15
)

# 4. TrainingArguments & Trainer 구성
###
training_args = TrainingArguments(
    output_dir="./checkpoints",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=100,
    evaluation_strategy="no",
    save_total_limit=2,
    remove_unused_columns=False,  # 중요: custom input 유지
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()







