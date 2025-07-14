# BertModelWithNumeric 통합 구조
import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel, Trainer
from transformers.models.bert.modeling_bert import BertEmbeddings, BertOnlyMLMHead, BertPooler
from transformers.modeling_outputs import SequenceClassifierOutput
from torch.utils.data import DataLoader
from transformers.tokenization_utils_base import BatchEncoding

class BertModelWithNumeric(BertPreTrainedModel):
    def __init__(self, config, gmt_embedder, numeric_cols):
        super().__init__(config)
        self.bert = BertModel(config)
        self.gmt_embedder = gmt_embedder  # GMTNumericEmbedder
        self.numeric_cols = numeric_cols

        self.d_model = config.hidden_size
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(self.d_model, eps=config.layer_norm_eps)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        df_numerics=None,
    ):
        # 1. 텍스트 임베딩 처리
        text_embeds = self.bert.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds
        )  # (batch_size, seq_len, hidden)

        # 2. 수치형 입력 임베딩
        if df_numerics is not None:
            numeric_embeds = self.gmt_embedder(df_numerics, self.numeric_cols)  # (batch_size, n_numeric, hidden)
            inputs_embeds = torch.cat([numeric_embeds, text_embeds], dim=1)

            if attention_mask is not None:
                B, T_text = attention_mask.shape
                T_numeric = numeric_embeds.shape[1]
                ones = torch.ones((B, T_numeric), dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat([ones, attention_mask], dim=1)
        else:
            inputs_embeds = text_embeds

        encoder_outputs = self.bert.encoder(
            inputs_embeds,
            attention_mask=attention_mask,
        )

        sequence_output = encoder_outputs[0]
        pooled_output = sequence_output[:, 0]

        return {
            "last_hidden_state": sequence_output,
            "pooled_output": pooled_output,
            "attention_mask": attention_mask
        }

class BertForPreTrainingWithNumeric(BertPreTrainedModel):
    def __init__(self, config, gmt_embedder, numeric_cols):
        super().__init__(config)
        self.bert_with_numeric = BertModelWithNumeric(config, gmt_embedder, numeric_cols)
        self.cls = BertOnlyMLMHead(config)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        labels=None,
        df_numerics=None,
    ):
        outputs = self.bert_with_numeric(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            df_numerics=df_numerics,
        )

        sequence_output = outputs["last_hidden_state"]
        prediction_scores = self.cls(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(prediction_scores.view(-1, prediction_scores.size(-1)), labels.view(-1))

        return {
            "loss": loss,
            "logits": prediction_scores,
            "hidden_states": outputs["last_hidden_state"],
        }
        
###############################################################################################
#BertForSequenceClassificationWithNumeric: 텍스트 + 수치형 입력을 받아 분류를 수행하는 통합 모델
#DataCollatorForNumericMLM: 수치형 입력이 포함된 MLM pretraining용 데이터 콜레이터
###############################################################################################
class BertForSequenceClassificationWithNumeric(BertPreTrainedModel):
    def __init__(self, config, gmt_embedder, numeric_cols):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert_with_numeric = BertModelWithNumeric(config, gmt_embedder, numeric_cols)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        labels=None,
        df_numerics=None,
    ):
        outputs = self.bert_with_numeric(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            df_numerics=df_numerics,
        )

        pooled_output = outputs["pooled_output"]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs["last_hidden_state"],
            attentions=None,
        )

class DataCollatorForNumericMLM:
    def __init__(self, tokenizer, gmt_embedder, numeric_cols, mlm_prob=0.15):
        self.tokenizer = tokenizer
        self.gmt_embedder = gmt_embedder
        self.numeric_cols = numeric_cols
        self.mlm_prob = mlm_prob

    def __call__(self, examples):
        text_inputs = self.tokenizer([ex["text"] for ex in examples], padding=True, truncation=True, return_tensors="pt")
        df_numerics = pd.DataFrame([{col: ex[col] for col in self.numeric_cols} for ex in examples])

        input_ids = text_inputs["input_ids"].clone()
        labels = input_ids.clone()
        probability_matrix = torch.full(input_ids.shape, self.mlm_prob)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
            for val in input_ids.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        input_ids[masked_indices] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        return {
            "input_ids": input_ids,
            "attention_mask": text_inputs["attention_mask"],
            "labels": labels,
            "df_numerics": df_numerics
        }

class TrainerWithNumeric(Trainer):
    def __init__(self, *args, numeric_cols=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.numeric_cols = numeric_cols

    def compute_loss(self, model, inputs, return_outputs=False):
        df_numerics = inputs.pop("df_numerics", None)
        outputs = model(**inputs, df_numerics=df_numerics)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
        return (loss, outputs) if return_outputs else loss


#from transformers import BertConfig, BertTokenizer
#config = BertConfig()
#tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
## 텍스트 토큰화
#text_batch = ["this is example", "another input"]
#encoding = tokenizer(text_batch, return_tensors="pt", padding=True)

## 수치형 입력 예시
#import pandas as pd
#df_numerics = pd.DataFrame({
#    "x1": [12.5, 7.2],
#    "x2": [101.3, 98.1]
#})
## GMT numeric embedder
#gmt_embedder = GMTNumericEmbedder(K=2, L=10, emb_dim=config.hidden_size, return_format="string")
#numeric_cols = ["x1", "x2"]

## 통합 모델
#model = BertModelWithNumeric(config, gmt_embedder, numeric_cols)
## forward
#output = model(
#    input_ids=encoding["input_ids"],
#    attention_mask=encoding["attention_mask"],
#    df_numerics=df_numerics
#)
#print(output["last_hidden_state"].shape)  # (batch, total_seq_len, hidden)


