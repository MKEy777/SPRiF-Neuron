# Diagnostic Analysis Figures — Manifest

每个子目录下的文件由对应的诊断分析脚本自动生成。
跑完脚本后将输出文件复制到对应目录即可。

---



## trajectory_analysis/ (Analysis 3.2)

**来源脚本**: `代码/experiments/trajectory_analysis/trajectory_analyze.py`

| 期望文件 | 格式 | 说明 |
|---------|------|------|
| `trajectory_comparison.png` | PNG | 慢状态连续性 vs 膜电位重置对比图 |
| `trajectory_data.npz` | NPZ | 逐时间步 slow/membrane/spike 数组 |

---

## impulse_analysis/ (Analysis 3.3)

**来源脚本**: `代码/experiments/impulse_analysis/impulse_analyze.py`

| 期望文件 | 格式 | 说明 |
|---------|------|------|
| `impulse_response_gallery.png` | PNG | 采样神经元冲激响应线图 |
| `frequency_response.png` | PNG | \|FFT\| 频域响应 |

| `impulse_kernel_stats.csv` | CSV | task, layer, neuron, alpha, omega, tau_effective |

---

## si_dms/ — active replacement for trajectory_visualization

SI-DMS replaces only the former `trajectory_visualization/` package. The separate `trajectory_analysis/`, impulse, and reset analyses remain active. Required outputs and evidence gates are defined in `si_dms/MANIFEST.md`. Archived trajectory-visualization files are provenance-only and cannot support reset arrows without recorded spikes and reset residuals.

---

## reset_analysis/ (Analysis 3.4)

**来源脚本**: `代码/experiments/reset_analysis/reset_analyze.py`

| 期望文件 | 格式 | 说明 |
|---------|------|------|
| `lambda_distribution.png` | PNG | λ 直方图 (含 λ=0 参考线) |
| `lambda_vs_firing_rate.png` | PNG | λ vs 发放率散点 |
| `lambda_vs_alpha.png` | PNG | λ vs α 散点 |
| `lambda_vs_omega.png` | PNG | λ vs ω 散点 |
| `lambda_stats.csv` | CSV | 全量: task, layer, neuron, α, ρ, ω, η₀, η₁, λ, firing_rate |
