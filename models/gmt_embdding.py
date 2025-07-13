# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 03:07:25 2025

@author: User
"""

import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertEmbeddings

class GMTBertEmbedding(nn.Module):
    def __init__(self, config, K=5, L=10, method="add"):
        super().__init__()
        self.bert_embeddings = BertEmbeddings(config)

        self.K = K
        self.L = L
        self.method = method
        d_model = config.hidden_size

        if method == "concat":
            assert d_model % 2 == 0
            d_sub = d_model // 2
            self.level_embedding = nn.Embedding(K, d_sub)
            self.sub_embedding = nn.Embedding(L, d_sub)
        else:  # "add"
            self.level_embedding = nn.Embedding(K, d_model)
            self.sub_embedding = nn.Embedding(L, d_model)

        self.LayerNorm = nn.LayerNorm(d_model, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        input_ids=None,         # usual BERT input_ids
        token_type_ids=None,
        position_ids=None,
        gmt_ids=None            # shape: (batch_size, seq_len, 2), i.e. (level, sub)
    ):
        # BERT 기본 embedding
        emb = self.bert_embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids
        )

        if gmt_ids is not None:
            # gmt_ids: (batch_size, seq_len, 2)
            level_ids = gmt_ids[..., 0]
            sub_ids = gmt_ids[..., 1]

            level_emb = self.level_embedding(level_ids)
            sub_emb = self.sub_embedding(sub_ids)

            if self.method == "concat":
                gmt_emb = torch.cat([level_emb, sub_emb], dim=-1)
            else:
                gmt_emb = level_emb + sub_emb

            # Add numeric embedding to BERT token embedding
            emb = emb + gmt_emb

        emb = self.LayerNorm(emb)
        emb = self.dropout(emb)
        return emb
