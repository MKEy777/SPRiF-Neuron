# Reset Direction Analysis (λ)

## 目的
分析 SPRiF 学习到的投影重置方向 λ_j——验证 λ 是否被真实学习（非平凡分布）、是否与发放率、谱参数存在有意义的相关性。

## 方法
1. 加载已训练的 SPRiF 模型（pSMNIST / GSC / ECG）
2. 提取每神经元 `lambda_reset` 值（`get_spectral_parameters()["lambda_reset"]`）
3. 在测试集上运行前向传播，记录每神经元发放率
4. 分析 λ 分布 + 与发放率/α/ω 的相关性

## 数据集
- **PS-MNIST**（2 层 [64, 256]）— PermutedMNIST 测试集
- **GSC**（1 层 [300]）— SpeechCommands 测试集
- **ECG / QTDB**（1 层 [36]）— QTDB 测试集

## 输出
- `lambda_distribution.png` — 每任务 λ 直方图，按层叠加，标注 λ=0 参考线
- `lambda_vs_firing_rate.png` — λ vs 发放率散点，按层着色（含相关系数 r）
- `lambda_vs_alpha.png` — λ vs α 散点（含 r）
- `lambda_vs_omega.png` — λ vs ω 散点（含 r）
- `lambda_stats.csv` — 完整数据表：task, layer, neuron, α, ρ, ω, η₀, η₁, λ, firing_rate

## 解读指南
- **λ 集中在 0 附近** → 投影重置退化为标量重置，C3 消融预期应无显著损失
- **λ 广泛分布（正/负皆有）** → 投影重置被积极学习，C3 消融应有性能下降
- **λ 与发放率正相关** → 高发放率神经元学习更大的 u₁ 维度重置幅度
- **λ 与 α 相关** → 重置策略与时域滤波特性耦合
- **λ 与 ω 相关** → 重置策略与频率选择性耦合
- **λ<0 的神经元** → 重置方向在 u₁ 维为负，可能与特定发放模式相关
