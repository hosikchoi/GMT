# utils/tokenizer.py
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer

def quantile_tokenizer(values, n_bins=20):
    est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    return est.fit_transform(values.values.reshape(-1, 1)).astype(int).flatten()

def histogram_tokenizer(values, n_bins=50):
    bins = np.linspace(0, 1, n_bins+1)
    scaled = (values - values.min()) / (values.max() - values.min())
    return np.digitize(scaled, bins) - 1

def fourier_embedding(values, max_power=2):
    # max_power=2 → periods = [10^0, 10^1, 10^2]
    output = []
    for i in range(max_power + 1):
        T = 10 ** i
        output.append(np.cos(2 * np.pi * values / T))
        output.append(np.sin(2 * np.pi * values / T))
    return np.stack(output, axis=1).T  # shape = (len(values), 2 * (max_power + 1))
