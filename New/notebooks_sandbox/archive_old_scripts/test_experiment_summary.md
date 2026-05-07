# Experiment Summary

## Model Configuration

| Parameter | Value |
|-----------|-------|
| model_size | medium |
| epochs | 500 |

## Training Results

| Metric | Value |
|--------|-------|
| Total Epochs | 3 |
| Final Train Loss | 0.080000 |
| Best Train Loss | 0.080000 |
| Final Val Loss | 0.090000 |
| Best Val Loss | 0.090000 |

## Distance Metrics

### Full Vector Distances

| Metric | Value |
|--------|-------|
| auto_regressive | 118.013184 |
| contractive | 1.406842 |
| cosine | 0.999294 |
| euclidean | 69.833794 |
| fisher_info_diff | 0.048389 |
| frobenius | 69.833794 |
| jacobian_norm | 69.833794 |
| jensen_shannon | 0.139885 |
| lwln | 1.102320 |
| manhattan | 2734.665527 |
| mape | 544.503784 |
| q_quantile | 0.937725 |
| wasserstein | 0.056265 |

### Layer-wise Distances

| Layer | Euclidean | Manhattan | Cosine |
|-------|-----------|-----------|--------|
| conv1_weights | 64.275009 | 2307.602051 | 0.993270 |
| conv1_bias | 7.715560 | 33.875679 | 1.358190 |
| conv2_weights | 26.190685 | 393.188202 | 1.013486 |
| conv2_bias | nan | nan | nan |
| fc_layer | nan | nan | nan |

## Topological Analysis

### Mapper Algorithm

| Metric | Value |
|--------|-------|
| Nodes | 16 |
| Edges | 13 |
| Intervals | 10 |
