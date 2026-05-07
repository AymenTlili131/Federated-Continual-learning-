## Full Vector Distances (2464 dimensions)

| Metric | Value |
|--------|-------|
| auto_regressive | 128.812180 |
| contractive | 0.238067 |
| cosine | 0.903065 |
| euclidean | 11.817343 |
| fisher_info_diff | 0.052225 |
| frobenius | 11.817343 |
| jacobian_norm | 11.817343 |
| jensen_shannon | 0.151898 |
| lwln | 0.694239 |
| manhattan | 404.052704 |
| mape | 6952197.656250 |
| q_quantile | 0.111587 |
| wasserstein | 0.123307 |

## Layer-wise Distances (5 subdistances)

| Layer | Euclidean | Manhattan | Cosine | Relative Diff | Mean Abs Diff |
|-------|-----------|-----------|--------|---------------|---------------|
| conv1_weights (2080w) | 8.410424 | 283.171356 | 0.847311 | 1.000445 | 0.136140 |
| conv1_bias (26w) | 1.262071 | 5.190858 | 1.154930 | 1.049924 | 0.199648 |
| conv2_weights (384w) | 8.204969 | 115.690521 | 0.992198 | 1.013927 | 0.323158 |
| conv2_bias (24w) | nan | nan | nan | nan | nan |
| fc_layer (-50w) | nan | nan | nan | nan | nan |

## Summary Statistics

| Statistic | Value |
|-----------|-------|
| max_layer_distance | 8.410424 |
| mean_full_euclidean | 11.817343 |
| mean_layerwise_euclidean | nan |
| min_layer_distance | 1.262071 |
| total_weights | 2464.000000 |