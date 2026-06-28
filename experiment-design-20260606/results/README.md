# SPRiF AAAI Submission — Results Directory

## 使用说明

所有实验跑完后，按以下方式填入结果：

### 数值结果 → JSON 文件

每个 `table*.json` 文件对应论文中一张表。打开文件，将 `"TBD"` 替换为实际数值。
- 所有数值保留 2–3 位有效数字
- `mean ± std` 格式：`"accuracy_mean": 0.952, "accuracy_std": 0.003`
- 运行多个 seed 后填入均值和标准差

### 图片结果 → figures/ 子目录

诊断分析脚本自动输出 PNG 到对应子目录。直接将图片文件复制到此处即可。
每个子目录下有 `MANIFEST.md` 列出期望的文件名。

### 训练日志 → training_logs/

将各实验的训练曲线、超参数、checkpoint 路径记录于此。

---

## 文件清单

| 文件 | 对应论文表 | 来源实验 |
|------|-----------|---------|
| `table1_main_results.json` | Table 1: Main Results | 主对比实验 (用户自行设计) |
| `table2_ablation_results.json` | Table 2: Mechanism Ablations | 消融 A/B/C × 3 数据集 |
| `table3_noise_robustness.json` | Table 3: Noise Robustness | 鲁棒性 R1 |
| `table4_sequence_noise.json` | Table 4: Sequence Length × Noise | 鲁棒性 R2 |
| `table5_frequency_selectivity.json` | Table 5: Frequency Selectivity | 鲁棒性 R3 |
| `figures/param_visualization/` | 参数分布图 (Fig X) | 诊断分析 3.1 + 3.5 |
| `figures/trajectory_analysis/` | 轨迹分析图 (Fig X) | 诊断分析 3.2 |
| `figures/impulse_analysis/` | 时间核分析图 (Fig X) | 诊断分析 3.3 |
| `figures/reset_analysis/` | Reset 方向分析图 (Fig X) | 诊断分析 3.4 |
| `training_logs/` | 训练曲线 / 超参 (Appendix) | 全部实验 |

---

## 快速填入流程

```
1. 跑完消融 A/B/C × 3 数据集
   → 打开 table2_ablation_results.json，填入 12 个 accuracy 值

2. 跑完鲁棒性 R1 (GSC + QTDB)
   → 打开 table3_noise_robustness.json，填入噪声下的 accuracy

3. 跑完诊断分析脚本
   → 将生成的 PNG/CSV/NPZ 复制到 figures/ 对应子目录

4. 跑完主对比 (你自行设计)
   → 打开 table1_main_results.json，填入所有 baseline 对比结果

5. 告诉我 "结果已填入"
   → 我会读取所有 JSON + 图片，开始写论文
```
