# BertModelWithNumeric
import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertEmbeddings

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
            # optional: prepend numeric embeddings before [CLS]
            inputs_embeds = torch.cat([numeric_embeds, text_embeds], dim=1)

            if attention_mask is not None:
                B, T_text = attention_mask.shape
                T_numeric = numeric_embeds.shape[1]
                ones = torch.ones((B, T_numeric), dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat([ones, attention_mask], dim=1)  # updated mask
        else:
            inputs_embeds = text_embeds

        # 3. BERT 인코더
        encoder_outputs = self.bert.encoder(
            inputs_embeds,
            attention_mask=attention_mask,
        )

        sequence_output = encoder_outputs[0]  # (batch_size, seq_len, hidden)
        pooled_output = sequence_output[:, 0]  # [CLS]

        return {
            "last_hidden_state": sequence_output,
            "pooled_output": pooled_output,
            "attention_mask": attention_mask
        }
