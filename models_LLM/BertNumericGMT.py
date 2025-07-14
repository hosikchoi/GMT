# BertModelWithNumeric 통합 구조
import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel, Trainer
from transformers.models.bert.modeling_bert import BertEmbeddings, BertOnlyMLMHead, BertPooler
from transformers.modeling_outputs import SequenceClassifierOutput
from torch.utils.data import DataLoader
from transformers.tokenization_utils_base import BatchEncoding
import pandas as pd
import torch.nn.functional as F

# Wasserstein 또는 MSE 기반 GMT 임베딩 정렬 loss

def compute_embedding_alignment_loss(embedding_layer, K, L, loss_type="mse"):
    # 임베딩 weight 추출: shape (K*L, dim)
    emb = embedding_layer.weight  # shared embedding: nn.Embedding(K*L, dim)
    device = emb.device

    # level, sub 인덱스 생성
    indices = torch.arange(K * L, device=device)
    level = indices // L
    sub = indices % L

    # 기준 거리 행렬 (예: level 거리 + sub 거리)
    with torch.no_grad():
        dist = (level.unsqueeze(1) - level.unsqueeze(0)).abs() + (sub.unsqueeze(1) - sub.unsqueeze(0)).abs()
        dist = dist.float() / dist.max()  # 정규화된 목표 거리

    # 실제 임베딩 거리 행렬
    diff = emb.unsqueeze(1) - emb.unsqueeze(0)  # (K*L, K*L, dim)
    dist_emb = torch.norm(diff, dim=-1)  # (K*L, K*L)

    if loss_type == "mse":
        loss = F.mse_loss(dist_emb, dist)
    elif loss_type == "wasserstein":
        loss = torch.mean(torch.abs(torch.cumsum(dist_emb, dim=0) - torch.cumsum(dist, dim=0)))
    else:
        raise ValueError("loss_type must be 'mse' or 'wasserstein'")

    return loss

#compute_embedding_alignment_loss(embedding_layer, K, L, loss_type="mse")
#alignment_loss = compute_embedding_alignment_loss(model.gmt_embedder.embedding, K=5, L=10)
#total_loss = task_loss + 0.1 * alignment_loss

# tokenizer 및 모델 저장 예시
def save_tokenizer_and_model(tokenizer, model, path="./checkpoints"):
    tokenizer.save_pretrained(f"{path}/tokenizer")
    model.save_pretrained(f"{path}/model")

# tokenizer 및 모델 로드 예시
def load_tokenizer_and_model(tokenizer_cls, model_cls, path, gmt_embedder, numeric_cols):
    tokenizer = tokenizer_cls.from_pretrained(f"{path}/tokenizer")
    model = model_cls.from_pretrained(f"{path}/model", gmt_embedder=gmt_embedder, numeric_cols=numeric_cols)
    return tokenizer, model
# GMT tokenizer 확장용 특수 토큰 생성기
def generate_gmt_tokens(K=5, L=10, add_special_tokens=True):
    special = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] if add_special_tokens else []
    gmt = [f"[level{k}][l{l}]" for k in range(K) for l in range(L)]
    return special + gmt

# tokenizer 확장 예시
def extend_tokenizer(tokenizer, K=5, L=10):
    gmt_tokens = generate_gmt_tokens(K, L, add_special_tokens=False)
    tokenizer.add_tokens(gmt_tokens, special_tokens=False)
    return tokenizer

