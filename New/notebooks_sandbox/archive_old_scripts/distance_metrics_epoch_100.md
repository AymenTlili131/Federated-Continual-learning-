## Full Vector Distances (2464 dimensions)

| Metric | Value |
|--------|-------|
| auto_regressive | 130.695114 |
| contractive | 0.235842 |
| cosine | 0.879047 |
| euclidean | 11.706896 |
| fisher_info_diff | 0.052992 |
| frobenius | 11.706896 |
| jacobian_norm | 11.706896 |
| jensen_shannon | 0.150299 |
| lwln | 0.683986 |
| manhattan | 398.085815 |
| mape | 6788711.718750 |
| q_quantile | 0.108051 |
| wasserstein | 0.129872 |

## Layer-wise Distances (5 subdistances)

| Layer | Euclidean | Manhattan | Cosine | Relative Diff | Mean Abs Diff |
|-------|-----------|-----------|--------|---------------|---------------|
| conv1_weights (2080w) | 8.272513 | 276.936340 | 0.804608 | 0.984040 | 0.133142 |
| conv1_bias (26w) | 1.187436 | 4.734951 | 0.843920 | 0.987835 | 0.182113 |
| conv2_weights (384w) | 8.197982 | 116.414513 | 1.011730 | 1.013063 | 0.325180 |
| conv2_bias (24w) | nan | nan | nan | nan | nan |
| fc_layer (-50w) | nan | nan | nan | nan | nan |

## Summary Statistics

| Statistic | Value |
|-----------|-------|
| max_layer_distance | 8.272513 |
| mean_full_euclidean | 11.706896 |
| mean_layerwise_euclidean | nan |
| min_layer_distance | 1.187436 |
| total_weights | 2464.000000 |