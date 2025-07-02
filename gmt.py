# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 19:21:00 2025

@author: Hosik
"""

# gmt_tokenizer.py
import numpy as np
import scipy.stats as stats

class GeneralMeasureTokenizer:
    def __init__(self, method='quantile', num_bins=20):
        self.method = method
        self.num_bins = num_bins
        self.bin_edges = None
        self.gauss_params = None

    def fit(self, x):
        x = np.asarray(x)
        if self.method == 'quantile':
            self.bin_edges = np.quantile(x, np.linspace(0, 1, self.num_bins + 1))
        elif self.method == 'gauss_rank':
            ranks = stats.rankdata(x)
            self.gauss_params = stats.norm.ppf((ranks - 0.5) / len(x))

    def transform(self, x):
        x = np.asarray(x)
        if self.method == 'quantile':
            return np.digitize(x, self.bin_edges[1:-1])
        elif self.method == 'gauss_rank':
            ranks = stats.rankdata(x)
            return stats.norm.ppf((ranks - 0.5) / len(x))

    def inverse_transform(self, tokens):
        if self.method == 'quantile':
            bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
            return bin_centers[tokens]
        elif self.method == 'gauss_rank':
            raise NotImplementedError("Inverse transform not defined for Gauss-rank")


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
