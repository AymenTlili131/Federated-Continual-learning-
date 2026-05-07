## Full Vector Distances (2464 dimensions)

| Metric | Value |
|--------|-------|
| auto_regressive | 133.414413 |
| contractive | 0.234031 |
| cosine | 0.857626 |
| euclidean | 11.616990 |
| fisher_info_diff | 0.054093 |
| frobenius | 11.616990 |
| jacobian_norm | 11.616990 |
| jensen_shannon | 0.162886 |
| lwln | 0.679413 |
| manhattan | 395.424377 |
| mape | 6863880.468750 |
| q_quantile | 0.106726 |
| wasserstein | 0.143697 |

## Layer-wise Distances (5 subdistances)

| Layer | Euclidean | Manhattan | Cosine | Relative Diff | Mean Abs Diff |
|-------|-----------|-----------|--------|---------------|---------------|
| conv1_weights (2080w) | 8.202261 | 274.870300 | 0.780814 | 0.975683 | 0.132149 |
| conv1_bias (26w) | 1.202653 | 4.709482 | 0.973014 | 1.000494 | 0.181134 |
| conv2_weights (384w) | 8.138243 | 115.844604 | 0.997965 | 1.005681 | 0.323588 |
| conv2_bias (24w) | nan | nan | nan | nan | nan |
| fc_layer (-50w) | nan | nan | nan | nan | nan |

## Summary Statistics

| Statistic | Value |
|-----------|-------|
| max_layer_distance | 8.202261 |
| mean_full_euclidean | 11.616990 |
| mean_layerwise_euclidean | nan |
| min_layer_distance | 1.202653 |
| total_weights | 2464.000000 |