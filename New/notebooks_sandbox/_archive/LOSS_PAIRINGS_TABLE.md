# Loss Function Pairings - Cross Table

## Individual Base Losses (Level 1)
- MSE
- MAE
- MAPE
- Quantile
- Sinkhorn
- FFT
- MelSpec
- JS (Jensen-Shannon)
- KL (KL Divergence)
- Frobenius
- LogNorm
- FIM (Fisher Information)
- AUTO (Autoregressive)

---

## Full Sequence Loss Pairings (Level 3)

| Main Loss ↓ / Reg Loss → | MSE | MAE | MAPE | Quantile | Sinkhorn | FFT | MelSpec | JS | KL | Frobenius | LogNorm | FIM |
|--------------------------|-----|-----|------|----------|----------|-----|---------|----|----|-----------|---------|-----|
| **MSE**                  | -   |     |      |          | ✅       |     |         |    |    | ✅        | ✅      |     |
| **MAE**                  |     | -   |      |          | ✅       |     |         |    |    | ✅        |         |     |
| **MAPE**                 |     |     | -    |          |          |     |         | ✅ |    |           |         |     |
| **Quantile**             |     |     |      | -        |          |     |         |    |    |           |         | ✅  |
| **Sinkhorn**             | ✅  | ✅  |      |          | -        |     |         |    | ✅ | ✅        |         |     |
| **FFT**                  |     |     |      |          |          | -   | ✅      |    |    |           |         |     |
| **MelSpec**              |     |     |      |          |          |     | -       |    |    |           |         |     |
| **JS**                   |     |     |      |          |          |     |         | -  |    |           |         |     |
| **KL**                   |     |     |      |          |          |     |         |    | -  |           |         |     |
| **Frobenius**            |     |     |      |          |          |     |         |    |    | -         |         |     |
| **LogNorm**              |     |     |      |          |          |     |         |    |    |           | -       |     |
| **FIM**                  |     |     |      |          |          |     |         |    |    |           |         | -   |

**Total Full Sequence Pairings: 12**

### Implemented Full Sequence Combinations:
1. ✅ MSE+0.05*Frobenius
2. ✅ MSE+0.1*LogNorm
3. ✅ MAPE+0.1*JS
4. ✅ Sinkhorn+0.15*KL
5. ✅ FFT+0.1*MelSpec
6. ✅ Quantile+0.05*FIM
7. ✅ MAE+0.05*Frobenius
8. ✅ Sinkhorn+0.1*MSE
9. ✅ Sinkhorn+0.1*MAE
10. ✅ Sinkhorn+0.1*Frobenius
11. ✅ MSE+0.15*Sinkhorn
12. ✅ MAE+0.15*Sinkhorn

---

## Layerwise Loss Pairings (Level 4)

| Main Loss ↓ / Reg Loss → | MSE | MAE | MAPE | Sinkhorn | FFT | MelSpec | JS | Frobenius | LogNorm | FIM |
|--------------------------|-----|-----|------|----------|-----|---------|----|-----------|---------|----|
| **MSE**                  | -   |     |      | ✅       |     |         |    | ✅        | ✅      |    |
| **MAE**                  |     | -   |      | ✅       |     |         |    |           |         | ✅ |
| **MAPE**                 |     |     | -    |          |     |         | ✅ |           |         |    |
| **Sinkhorn**             | ✅  | ✅  |      | -        |     |         |    | ✅        |         |    |
| **FFT**                  |     |     |      |          | -   | ✅      |    |           |         |    |
| **MelSpec**              |     |     |      |          |     | -       |    |           |         |    |
| **JS**                   |     |     |      |          |     |         | -  |           |         |    |
| **Frobenius**            |     |     |      |          |     |         |    | -         |         |    |
| **LogNorm**              |     |     |      |          |     |         |    |           | -       |    |
| **FIM**                  |     |     |      |          |     |         |    |           |         | -  |

**Total Layerwise Pairings: 10**

### Implemented Layerwise Combinations:
1. ✅ LW_MSE+0.05*LW_Frobenius
2. ✅ LW_MSE+0.1*LW_LogNorm
3. ✅ LW_MAPE+0.1*LW_JS
4. ✅ LW_MAE+0.05*LW_FIM
5. ✅ LW_FFT+0.1*LW_MelSpec
6. ✅ LW_Sinkhorn+0.1*LW_MSE
7. ✅ LW_Sinkhorn+0.1*LW_MAE
8. ✅ LW_Sinkhorn+0.1*LW_Frobenius
9. ✅ LW_MSE+0.15*LW_Sinkhorn
10. ✅ LW_MAE+0.15*LW_Sinkhorn

---

## Summary Statistics

### Total Losses Available: 48

**By Level:**
- Level 1 (Individual): 13 losses
- Level 2 (Layerwise): 13 losses (12 + LWLN)
- Level 3 (Regularized Full): 12 pairings
- Level 4 (Regularized Layerwise): 10 pairings
- Level 5 (Mixed): 0 (removed due to NaN issues)

**Experiment Sequence: 33 losses**
- 6 individual losses
- 5 layerwise losses
- 12 regularized full pairings
- 10 regularized layerwise pairings

### Sinkhorn-Based Experiments: 11 total
**Full Sequence:**
1. Sinkhorn (standalone)
2. Sinkhorn+0.15*KL
3. Sinkhorn+0.1*MSE
4. Sinkhorn+0.1*MAE
5. Sinkhorn+0.1*Frobenius
6. MSE+0.15*Sinkhorn
7. MAE+0.15*Sinkhorn

**Layerwise:**
8. LW_Sinkhorn (standalone)
9. LW_Sinkhorn+0.1*LW_MSE
10. LW_Sinkhorn+0.1*LW_MAE
11. LW_Sinkhorn+0.1*LW_Frobenius
12. LW_MSE+0.15*LW_Sinkhorn
13. LW_MAE+0.15*LW_Sinkhorn

---

## Possible Future Pairings (Not Yet Implemented)

### Full Sequence Candidates:
- MSE + MAE
- MSE + MAPE
- MSE + FFT
- MAE + MAPE
- MAE + FFT
- MAPE + Frobenius
- FFT + Frobenius
- Sinkhorn + FFT
- Sinkhorn + MAPE
- KL + JS
- ... (many more possible)

### Layerwise Candidates:
- LW_MSE + LW_MAE
- LW_MSE + LW_FFT
- LW_Sinkhorn + LW_FFT
- LW_Sinkhorn + LW_MAPE
- ... (many more possible)

**Note:** Current implementation focuses on theoretically motivated pairings based on:
- Complementary properties (e.g., MSE + Frobenius for norm regularization)
- Different domains (e.g., FFT + MelSpec for frequency analysis)
- Distribution matching (e.g., Sinkhorn + KL for optimal transport + divergence)

---

## Device Compatibility

✅ **All losses now handle device placement correctly**
- Sinkhorn loss moves tensors to CPU for computation (geomloss limitation)
- Results moved back to original device for gradient flow
- All other losses work on GPU/CPU automatically

---

**Last Updated:** After adding 5 new Sinkhorn combinations (full + layerwise)
