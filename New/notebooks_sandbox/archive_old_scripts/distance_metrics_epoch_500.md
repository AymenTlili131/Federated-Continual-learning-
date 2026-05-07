## Full Vector Distances (2464 dimensions)

| Metric | Value |
|--------|-------|
| auto_regressive | 133.505920 |
| contractive | 0.233831 |
| cosine | 0.853112 |
| euclidean | 11.607055 |
| fisher_info_diff | 0.054130 |
| frobenius | 11.607055 |
| jacobian_norm | 11.607055 |
| jensen_shannon | 0.165722 |
| lwln | 0.679259 |
| manhattan | 395.334564 |
| mape | 6764010.937500 |
| q_quantile | 0.106643 |
| wasserstein | 0.145060 |

## Layer-wise Distances (5 subdistances)

| Layer | Euclidean | Manhattan | Cosine | Relative Diff | Mean Abs Diff |
|-------|-----------|-----------|--------|---------------|---------------|
| conv1_weights (2080w) | 8.201742 | 275.018097 | 0.780531 | 0.975622 | 0.132220 |
| conv1_bias (26w) | 1.202316 | 4.714643 | 0.977385 | 1.000213 | 0.181332 |
| conv2_weights (384w) | 8.124629 | 115.601822 | 0.984036 | 1.003999 | 0.322910 |
| conv2_bias (24w) | nan | nan | nan | nan | nan |
| fc_layer (-50w) | nan | nan | nan | nan | nan |

## Summary Statistics

| Statistic | Value |
|-----------|-------|
| max_layer_distance | 8.201742 |
| mean_full_euclidean | 11.607055 |
| mean_layerwise_euclidean | nan |
| min_layer_distance | 1.202316 |
| total_weights | 2464.000000 |