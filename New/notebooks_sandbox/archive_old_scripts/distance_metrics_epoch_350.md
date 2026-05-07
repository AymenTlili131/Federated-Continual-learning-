## Full Vector Distances (2464 dimensions)

| Metric | Value |
|--------|-------|
| auto_regressive | 133.252823 |
| contractive | 0.233859 |
| cosine | 0.852942 |
| euclidean | 11.608456 |
| fisher_info_diff | 0.054028 |
| frobenius | 11.608456 |
| jacobian_norm | 11.608456 |
| jensen_shannon | 0.162881 |
| lwln | 0.680718 |
| manhattan | 396.183777 |
| mape | 6782575.781250 |
| q_quantile | 0.107846 |
| wasserstein | 0.142549 |

## Layer-wise Distances (5 subdistances)

| Layer | Euclidean | Manhattan | Cosine | Relative Diff | Mean Abs Diff |
|-------|-----------|-----------|--------|---------------|---------------|
| conv1_weights (2080w) | 8.220504 | 275.745972 | 0.790171 | 0.977853 | 0.132570 |
| conv1_bias (26w) | 1.197105 | 4.724418 | 0.905811 | 0.995878 | 0.181708 |
| conv2_weights (384w) | 8.108421 | 115.713394 | 0.964222 | 1.001996 | 0.323222 |
| conv2_bias (24w) | nan | nan | nan | nan | nan |
| fc_layer (-50w) | nan | nan | nan | nan | nan |

## Summary Statistics

| Statistic | Value |
|-----------|-------|
| max_layer_distance | 8.220504 |
| mean_full_euclidean | 11.608456 |
| mean_layerwise_euclidean | nan |
| min_layer_distance | 1.197105 |
| total_weights | 2464.000000 |