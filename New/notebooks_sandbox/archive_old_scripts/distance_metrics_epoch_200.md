## Full Vector Distances (2464 dimensions)

| Metric | Value |
|--------|-------|
| auto_regressive | 132.025955 |
| contractive | 0.234405 |
| cosine | 0.859351 |
| euclidean | 11.635562 |
| fisher_info_diff | 0.053530 |
| frobenius | 11.635562 |
| jacobian_norm | 11.635562 |
| jensen_shannon | 0.148594 |
| lwln | 0.681684 |
| manhattan | 396.745575 |
| mape | 6957775.781250 |
| q_quantile | 0.108982 |
| wasserstein | 0.135532 |

## Layer-wise Distances (5 subdistances)

| Layer | Euclidean | Manhattan | Cosine | Relative Diff | Mean Abs Diff |
|-------|-----------|-----------|--------|---------------|---------------|
| conv1_weights (2080w) | 8.233978 | 276.215088 | 0.794550 | 0.979456 | 0.132796 |
| conv1_bias (26w) | 1.177742 | 4.546450 | 0.794166 | 0.979770 | 0.174863 |
| conv2_weights (384w) | 8.136390 | 115.984055 | 0.968064 | 1.005452 | 0.323978 |
| conv2_bias (24w) | nan | nan | nan | nan | nan |
| fc_layer (-50w) | nan | nan | nan | nan | nan |

## Summary Statistics

| Statistic | Value |
|-----------|-------|
| max_layer_distance | 8.233978 |
| mean_full_euclidean | 11.635562 |
| mean_layerwise_euclidean | nan |
| min_layer_distance | 1.177742 |
| total_weights | 2464.000000 |