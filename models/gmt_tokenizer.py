# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 19:21:00 2025

@author: Hosik
"""

# gmt_tokenizer.py
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from transformers import PreTrainedTokenizer

class GeneralMeasureTokenizer:
    def __init__(self, method='quantile', num_bins=20):
        self.method = method
        self.num_bins = num_bins
        self.bin_edges = None
        self.gauss_params = None
        self.fitted = False

    def fit(self, x):
        x = np.asarray(x)
        if self.method == 'quantile':
            self.bin_edges = np.quantile(x, np.linspace(0, 1, self.num_bins + 1))
        elif self.method == 'gauss_rank':
            ranks = stats.rankdata(x)
            self.gauss_params = stats.norm.ppf((ranks - 0.5) / len(x))
        self.fitted = True

    def transform(self, x):
        assert self.fitted, "Tokenizer must be fit first."
        x = np.asarray(x)
        if self.method == 'quantile':
            return np.digitize(x, self.bin_edges[1:-1])
        elif self.method == 'gauss_rank':
            ranks = stats.rankdata(x)
            return stats.norm.ppf((ranks - 0.5) / len(x))

    def inverse_transform(self, tokens):
        if not self.fitted:
            raise RuntimeError("Must call fit() before inverse_transform().")        
        if self.method == 'quantile':
            bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
            return bin_centers[tokens]
        elif self.method == 'gauss_rank':
            raise NotImplementedError("Inverse transform not defined for Gauss-rank")
            
    def token2value(self, token):
        return self.inverse_transform(np.array([token]))[0]
    
    def value2token(self, value):
        return self.transform(np.array([value]))[0]
    
    def decode_row(self, token_row):
        return self.inverse_transform(np.array(token_row))

    def visualize_bins(self):
        if self.method == 'quantile':
            plt.figure(figsize=(8, 2))
            plt.hist(self.bin_edges[:-1], bins=self.bin_edges, edgecolor='k')
            plt.title("Quantile-based Bins")
            plt.show()
        else:
            raise NotImplementedError("Visualization only supported for quantile mode")
    
def convert_tokens_to_values(tokens, gmt_tokenizer):
    values = []
    for token in tokens:
        if token.startswith("<num_"):
            token_id = int(token[5:-1])
            values.append(gmt_tokenizer.token2value(token_id))
        else:
            values.append(token)
    return values

def level_subbin_token(level_idx, subbin_idx, unit_prefix):
    return f"[level{level_idx:02d}][{unit_prefix}{subbin_idx}]"

def tokenize_numerical_column(values, gmt_tokenizer, unit_prefix="l"):
    gmt_tokenizer.fit(values)
    bins = gmt_tokenizer.transform(values)
    tokens = [level_subbin_token(level, level % gmt_tokenizer.num_bins, unit_prefix) for level in bins]
    return tokens

class HuggingfaceGeneralMeasureTokenizer(PreTrainedTokenizer):
    def __init__(self, gmt_tokenizer, **kwargs):
        super().__init__(**kwargs)
        self.gmt = gmt_tokenizer
        self.vocab = {}
        self.ids_to_tokens = {}
        self.counter = 0

    def _tokenize(self, text):
        try:
            x = float(text)
            token = f"<num_{self.gmt.value2token(x):02d}>"
            return [token]
        except ValueError:
            return text.split()
    
    def _convert_token_to_id(self, token):
        if token not in self.vocab:
            self.vocab[token] = self.counter
            self.ids_to_tokens[self.counter] = token
            self.counter += 1
        return self.vocab[token]
    
    def _convert_id_to_token(self, index):
        return self.ids_to_tokens.get(index, "<unk>")
        #def _convert_token_to_id(self, token):
        #    return hash(token) % 10000

    #def _convert_id_to_token(self, index):
    #    return f"<num_{index}>"

    def convert_tokens_to_string(self, tokens):
        return ' '.join(tokens)





