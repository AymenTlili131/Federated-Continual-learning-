## Full Vector Distances (2464 dimensions)

| Metric | Value |
|--------|-------|
| auto_regressive | 133.470703 |
| contractive | 0.233869 |
| cosine | 0.853914 |
| euclidean | 11.608931 |
| fisher_info_diff | 0.054116 |
| frobenius | 11.608931 |
| jacobian_norm | 11.608931 |
| jensen_shannon | 0.163509 |
| lwln | 0.679479 |
| manhattan | 395.462280 |
| mape | 6798389.062500 |
| q_quantile | 0.105898 |
| wasserstein | 0.144679 |

## Layer-wise Distances (5 subdistances)

| Layer | Euclidean | Manhattan | Cosine | Relative Diff | Mean Abs Diff |
|-------|-----------|-----------|--------|---------------|---------------|
| conv1_weights (2080w) | 8.202004 | 275.107544 | 0.780677 | 0.975653 | 0.132263 |
| conv1_bias (26w) | 1.201431 | 4.710514 | 0.963951 | 0.999477 | 0.181174 |
| conv2_weights (384w) | 8.127174 | 115.644211 | 0.986360 | 1.004313 | 0.323029 |
| conv2_bias (24w) | nan | nan | nan | nan | nan |
| fc_layer (-50w) | nan | nan | nan | nan | nan |

## Summary Statistics

| Statistic | Value |
|-----------|-------|
| max_layer_distance | 8.202004 |
| mean_full_euclidean | 11.608931 |
| mean_layerwise_euclidean | nan |
| min_layer_distance | 1.201431 |
| total_weights | 2464.000000 |