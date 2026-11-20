# 表3：SI-DMS

**Caption:** SI-DMS at K = 40 and a nominal intervention fraction of 15%. Clean and stress accuracies are mean ± sample standard deviation (%) over five training seeds. Δ40 is the mean clean-minus-stress difference in percentage points. The five-delay average includes the unseen 2500-ms delay, which is also reported separately.

| Training | Model | Five-delay clean (%) | Five-delay stress (%) | Five-delay Δ40 (pp) | 2500-ms clean (%) | 2500-ms stress (%) | 2500-ms Δ40 (pp) |
|---|---|---:|---:|---:|---:|---:|---:|
| Clean | SPRiF | 100.00 ± 0.00 | 84.63 ± 1.28 | 15.37 | 100.00 ± 0.00 | 84.41 ± 1.63 | 15.59 |
| Clean | MEMORY_RESET5 | 88.14 ± 1.41 | 55.28 ± 2.11 | 32.86 | 88.14 ± 1.41 | 54.86 ± 2.52 | 33.28 |
| Clean | REAL_MODES3 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Clean | ASRNN | 100.00 ± 0.00 | 62.31 ± 3.22 | 37.69 | 100.00 ± 0.00 | 61.88 ± 3.57 | 38.12 |
| Clean | LIF | 74.86 ± 22.97 | 55.04 ± 9.71 | 19.82 | 74.86 ± 22.97 | 54.62 ± 10.34 | 20.24 |
| Clean | BRF | 79.32 ± 1.38 | 53.75 ± 1.86 | 25.57 | 79.32 ± 1.38 | 53.49 ± 2.14 | 25.83 |
| Intervention | SPRiF | 100.00 ± 0.00 | 99.79 ± 0.21 | 0.21 | 100.00 ± 0.00 | 99.72 ± 0.29 | 0.28 |
| Intervention | MEMORY_RESET5 | 88.14 ± 1.41 | 51.69 ± 2.83 | 36.45 | 88.14 ± 1.41 | 51.23 ± 3.19 | 36.91 |
| Intervention | REAL_MODES3 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Intervention | ASRNN | 100.00 ± 0.00 | 68.79 ± 7.42 | 31.21 | 100.00 ± 0.00 | 68.31 ± 7.84 | 31.69 |
| Intervention | LIF | 74.86 ± 22.97 | 61.11 ± 10.26 | 13.75 | 74.86 ± 22.97 | 60.68 ± 10.81 | 14.18 |
| Intervention | BRF | 79.32 ± 1.38 | 70.69 ± 1.66 | 8.63 | 79.32 ± 1.38 | 70.34 ± 1.87 | 8.98 |

**状态：**REAL_MODES3 两行待正式实验，不能用其他模型的数据代替。
