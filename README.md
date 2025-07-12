# General Measure Tokens (GMT)
This repository provides the official implementation of General Measure Tokens (GMT), 
a hierarchical and unit-aware tokenization framework designed for encoding numerical 
values in tabular data. GMT improves downstream performance in tasks such as imputation, 
interpolation, and instruction-based generation by preserving ordinal structure and unit semantics.

## Overview
Traditional numeric tokenization approaches suffer from issues like:
- Loss of ordinal structure (e.g., treating 50 and 51 as entirely distinct tokens)
- Lack of unit-awareness (e.g., height vs. weight)
- Poor generalization in interpolation and reasoning

GMT addresses these issues by introducing **[level-k][u-l]** style tokens, where:
- `level-k` denotes coarse-grained scale (e.g., quantile level)
- `u-l` denotes unit-specific fine-grained binning

The proposed scheme enables:
- Compact and interpretable tokenization
- Wasserstein-based proximity losses
- Embedding regularization for ordinal alignment
- Compatibility with masked language modeling (MLM)

## Project Structure
GMT/
├── gmt/ # Core GMT module (tokenizer, embeddings)
|
├── scripts/ # Training and evaluation scripts
|
├── data/ # Sample or synthetic datasets
|
├── run_all.py # Integrated experiment pipeline
|
└── README.md
