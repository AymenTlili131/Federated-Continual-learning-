## Full Vector Distances (2464 dimensions)

| Metric | Value |
|--------|-------|
| auto_regressive | 132.969131 |
| contractive | 0.233960 |
| cosine | 0.854193 |
| euclidean | 11.613446 |
| fisher_info_diff | 0.053913 |
| frobenius | 11.613446 |
| jacobian_norm | 11.613446 |
| jensen_shannon | 0.157133 |
| lwln | 0.682923 |
| manhattan | 397.466980 |
| mape | 6768519.531250 |
| q_quantile | 0.110385 |
| wasserstein | 0.140017 |

## Layer-wise Distances (5 subdistances)

| Layer | Euclidean | Manhattan | Cosine | Relative Diff | Mean Abs Diff |
|-------|-----------|-----------|--------|---------------|---------------|
| conv1_weights (2080w) | 8.244126 | 277.179108 | 0.801453 | 0.980663 | 0.133259 |
| conv1_bias (26w) | 1.215492 | 4.706989 | 1.111730 | 1.011174 | 0.181038 |
| conv2_weights (384w) | 8.088825 | 115.580902 | 0.940255 | 0.999574 | 0.322852 |
| conv2_bias (24w) | nan | nan | nan | nan | nan |
| fc_layer (-50w) | nan | nan | nan | nan | nan |

## Summary Statistics

| Statistic | Value |
|-----------|-------|
| max_layer_distance | 8.244126 |
| mean_full_euclidean | 11.613446 |
| mean_layerwise_euclidean | nan |
| min_layer_distance | 1.215492 |
| total_weights | 2464.000000 |