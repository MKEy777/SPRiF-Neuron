# SPRiF AAAI Submission — Results Directory

## 当前进度

| 文件 | 状态 | 更新日期 |
|------|------|---------|
| `table1_main_results.json` | ✅ 已填入 (5 datasets, 7 baselines + SPRiF) | 2026-07-01 |
| `table2_ablation_results.json` | ✅ 已填入 (PS-MNIST, GSC, ECG × 4 variants) | 2026-07-01 |


## 数据来源

主实验和消融实验结果来自 PDF 文件 `实验_SPRiF.pdf`（日期: 2026-07-01）。

> **注**: PDF 中的 "ECG" 即论文中的 QTDB 数据集（实验代码中使用的名称是 ECG）。

## Baseline 方法参考文献

| 简称 | 论文标题 | 发表venue | 出现数据集 |
|------|---------|----------|-----------|
| **GLIF** | GLIF: A Unified Gated Leaky Integrate-and-Fire Neuron for Spiking Neural Networks | NeurIPS 2022 | S-MNIST, PS-MNIST |
| **ASRNN** | Accurate and Efficient Time-Domain Classification with Adaptive Spiking Recurrent Neural Networks | Nat. Mach. Intell. 2021 | S-MNIST, PS-MNIST, QTDB, GSC, SHD |
| **DH-SNN** | Temporal Dendritic Heterogeneity Incorporated with Spiking Neural Networks for Learning Multi-Timescale Dynamics | Nat. Commun. 2024 | S-MNIST, PS-MNIST, QTDB, GSC, SHD |
| **BHRF** | Balanced Resonate-and-Fire Neurons | ICLR 2024 | S-MNIST, PS-MNIST, QTDB, SHD |
| **TC-LIF** | TC-LIF: A Two-Compartment Spiking Neuron Model for Long-Term Sequential Modelling | AAAI 2024 | S-MNIST, PS-MNIST, GSC, SHD |
| **SE-adLIF** | Advancing Spatio-Temporal Processing through Adaptation in Spiking Neural Networks | Nat. Commun. 2025 | QTDB |
| **SNN-SFA** | Spike Frequency Adaptation Supports Network Computations on Temporally Dispersed Information | eLife 2021 | GSC |
| **RadLIF** | A Surrogate Gradient Spiking Baseline for Speech Command Recognition | Front. Neurosci. 2022 | GSC |
| **Heterogeneous SNN** | Neural Heterogeneity Promotes Robust Learning | Nat. Commun. 2021 | SHD |
| **MPS-SNN** | Rethinking Spiking Neural Networks from an Ensemble Learning Perspective | ICLR 2025 | SHD |
| **DGN** | A Brain-Inspired Gating Mechanism Unlocks Robust Computation in Spiking Neural Networks | ICLR 2026 | SHD |

> **注**: SHD 表中原始数据标记为 "BRF"，已统一为 BHRF（同一方法，ICLR 2024）。

---

## 关键结果摘要

### 主实验 (Table 1)

| Dataset | SPRiF Acc (%) | Best Baseline | Baseline Acc (%) | ΔAcc |
|---------|--------------|---------------|------------------|------|
| S-MNIST | **99.28** | TC-LIF | 99.20 | +0.08 |
| PS-MNIST | **95.86** | BHRF | 95.20 | +0.66 |
| QTDB (ECG) | **88.43** | BHRF | 87.00 | +1.43 |
| GSC | 94.55 | TC-LIF | **94.84** | -0.29 |
| SHD | 91.52 | BRF | **91.70** | -0.18 |

**注**：GSC 和 SHD 上 SPRiF 不是最佳，但参数量显著更小。
- GSC: SPRiF 0.13M vs TC-LIF 0.20M（少 35%）
- SHD: SPRiF 0.05M vs BRF 0.11M（少 55%）

### 消融实验 (Table 2)

| Dataset | Full | a (ω=0) | b (merged) | c (scalar) |
|---------|------|---------|------------|------------|
| PS-MNIST | 95.82 | 92.29 (-3.53) | 93.44 (-2.38) | 95.32 (-0.50) |
| GSC | 94.55 | 93.83 (-0.72) | 90.05 (-4.50) | 94.22 (-0.33) |
| QTDB (ECG) | 88.43 | 87.78 (-0.65) | 83.39 (-5.04) | 87.40 (-1.03) |

**观察**：消融 b（合并 slow+fast）在三个数据集上影响最大，说明双时间尺度机制是核心贡献。

## 使用说明

### 数值结果 → JSON 文件

每个 `table*.json` 文件对应论文中一张表。打开文件，将 `"TBD"` 替换为实际数值。
- 所有数值保留 2–3 位有效数字
- `mean ± std` 格式：`"accuracy_mean": 0.952, "accuracy_std": 0.003`
- 运行多个 seed 后填入均值和标准差

### 图片结果 → figures/ 子目录

诊断分析脚本自动输出 PNG 到对应子目录。直接将图片文件复制到此处即可。
每个子目录下有 `MANIFEST.md` 列出期望的文件名。

---

## 文件清单

| 文件 | 对应论文表 | 来源实验 |
|------|-----------|---------|
| `table1_main_results.json` | Table 1: Main Results | 主对比实验 (5 datasets) |
| `table2_ablation_results.json` | Table 2: Mechanism Ablations | 消融 A/B/C × 3 数据集 |

| `figures/param_visualization/` | 参数分布图 (Fig X) | 诊断分析 3.1 + 3.5 |
| `figures/trajectory_analysis/` | 轨迹分析图 (Fig X) | 诊断分析 3.2 |
| `figures/impulse_analysis/` | 时间核分析图 (Fig X) | 诊断分析 3.3 |
| `figures/reset_analysis/` | Reset 方向分析图 (Fig X) | 诊断分析 3.4 |


---

## 快速填入流程

```
1. ✅ 主对比实验 → table1_main_results.json 已填入

2. ✅ 消融 A/B/C × 3 数据集 → table2_ablation_results.json 已填入



6. 跑完诊断分析脚本
   → 将生成的 PNG/CSV/NPZ 复制到 figures/ 对应子目录

7. 告诉我 "结果已填入"
   → 我会读取所有 JSON + 图片，开始写论文
```

---

## SI-DMS evidence replacement (2026-07-11)

SI-DMS replaces **only** the former `trajectory_visualization` experiment. Main benchmark results, standard mechanism ablations, `trajectory_analysis`, impulse analysis, reset analysis, frequency selectivity, noise robustness, and sequence noise remain active.

- Active plan: `../si-dms-experiment-plan.md`
- Result templates: `si_dms/`
- Figure contract: `figures/si_dms/MANIFEST.md`
- Archived predecessor: `../legacy/trajectory_visualization/`

The archive is provenance-only. It must not support a reset claim when the recorded trajectory has no real spike or no nonzero reset residual.
