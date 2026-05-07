## Full Vector Distances (2464 dimensions)

| Metric | Value |
|--------|-------|
| auto_regressive | 132.689484 |
| contractive | 0.234697 |
| cosine | 0.869404 |
| euclidean | 11.650055 |
| fisher_info_diff | 0.053799 |
| frobenius | 11.650055 |
| jacobian_norm | 11.650055 |
| jensen_shannon | 0.154599 |
| lwln | 0.683210 |
| manhattan | 397.634216 |
| mape | 6556650.781250 |
| q_quantile | 0.109129 |
| wasserstein | 0.138516 |

## Layer-wise Distances (5 subdistances)

| Layer | Euclidean | Manhattan | Cosine | Relative Diff | Mean Abs Diff |
|-------|-----------|-----------|--------|---------------|---------------|
| conv1_weights (2080w) | 8.256649 | 276.988586 | 0.807276 | 0.982153 | 0.133168 |
| conv1_bias (26w) | 1.227503 | 4.836850 | 1.184815 | 1.021167 | 0.186033 |
| conv2_weights (384w) | 8.126793 | 115.808731 | 0.972555 | 1.004266 | 0.323488 |
| conv2_bias (24w) | nan | nan | nan | nan | nan |
| fc_layer (-50w) | nan | nan | nan | nan | nan |

## Summary Statistics

| Statistic | Value |
|-----------|-------|
| max_layer_distance | 8.256649 |
| mean_full_euclidean | 11.650055 |
| mean_layerwise_euclidean | nan |
| min_layer_distance | 1.227503 |
| total_weights | 2464.000000 |