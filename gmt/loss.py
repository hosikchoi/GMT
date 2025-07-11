# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 19:45:44 2025

@author: Hosik
"""

# losses.py
import torch
import torch.nn.functional as F

def ntl_mse_loss(logits, labels, num_token_values):
    probs = F.softmax(logits, dim=-1)
    expected_value = torch.sum(probs * num_token_values, dim=-1)
    true_value = num_token_values[labels]
    return F.mse_loss(expected_value, true_value)

def ntl_wasserstein_loss(logits, labels, num_token_values):
    probs = F.softmax(logits, dim=-1)
    one_hot = F.one_hot(labels, num_classes=num_token_values.shape[0]).float()
    cdf_pred = torch.cumsum(probs, dim=-1)
    cdf_true = torch.cumsum(one_hot, dim=-1)
    return torch.mean(torch.abs(cdf_pred - cdf_true))

def masked_wasserstein_loss(logits, target, mask):
    """
    logits: (B, T, V)
    target: (B, T)
    mask: (B, T) boolean
    """
    B, T, V = logits.shape
    probs = F.softmax(logits, dim=-1)
    target_onehot = F.one_hot(target, V).float()
    cdf_pred = torch.cumsum(probs, dim=-1)
    cdf_true = torch.cumsum(target_onehot, dim=-1)
    w1 = torch.abs(cdf_pred - cdf_true).sum(dim=-1)
    return (w1 * mask).sum() / mask.sum()
