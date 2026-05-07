## Full Vector Distances (2464 dimensions)

| Metric | Value |
|--------|-------|
| auto_regressive | 131.751007 |
| contractive | 0.234871 |
| cosine | 0.867350 |
| euclidean | 11.658704 |
| fisher_info_diff | 0.053418 |
| frobenius | 11.658704 |
| jacobian_norm | 11.658704 |
| jensen_shannon | 0.151128 |
| lwln | 0.686088 |
| manhattan | 399.309021 |
| mape | 6892472.656250 |
| q_quantile | 0.109340 |
| wasserstein | 0.133119 |

## Layer-wise Distances (5 subdistances)

| Layer | Euclidean | Manhattan | Cosine | Relative Diff | Mean Abs Diff |
|-------|-----------|-----------|--------|---------------|---------------|
| conv1_weights (2080w) | 8.289617 | 278.371582 | 0.817839 | 0.986074 | 0.133832 |
| conv1_bias (26w) | 1.216817 | 4.840724 | 1.027849 | 1.012277 | 0.186182 |
| conv2_weights (384w) | 8.107218 | 116.096741 | 0.946778 | 1.001847 | 0.324293 |
| conv2_bias (24w) | nan | nan | nan | nan | nan |
| fc_layer (-50w) | nan | nan | nan | nan | nan |

## Summary Statistics

| Statistic | Value |
|-----------|-------|
| max_layer_distance | 8.289617 |
| mean_full_euclidean | 11.658704 |
| mean_layerwise_euclidean | nan |
| min_layer_distance | 1.216817 |
| total_weights | 2464.000000 |