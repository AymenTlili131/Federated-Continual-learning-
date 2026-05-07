## Full Vector Distances (2464 dimensions)

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

## Layer-wise Distances (5 subdistances)

| Layer | Euclidean | Manhattan | Cosine | Relative Diff | Mean Abs Diff |
|-------|-----------|-----------|--------|---------------|---------------|
| conv1_weights (2080w) | 64.275009 | 2307.602051 | 0.993270 | 1.393383 | 1.109424 |
| conv1_bias (26w) | 7.715560 | 33.875679 | 1.358190 | 1.857057 | 1.302911 |
| conv2_weights (384w) | 26.190685 | 393.188202 | 1.013486 | 1.392051 | 1.098291 |
| conv2_bias (24w) | nan | nan | nan | nan | nan |
| fc_layer (-50w) | nan | nan | nan | nan | nan |

## Summary Statistics

| Statistic | Value |
|-----------|-------|
| max_layer_distance | 64.275009 |
| mean_full_euclidean | 69.833794 |
| mean_layerwise_euclidean | nan |
| min_layer_distance | 7.715560 |
| total_weights | 2464.000000 |