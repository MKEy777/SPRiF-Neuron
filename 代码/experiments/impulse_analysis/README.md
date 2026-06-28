# Effective Temporal Kernel Analysis

## 目的
展示 SPRiF 学习到的多样化时间滤波器——不同于 LIF 的单一指数衰减核，SPRiF 的谱参数化让每个神经元学到不同的冲激响应（快/慢衰减、振荡）。

## 方法
1. 加载已训练的 SPRiF 模型（pSMNIST / GSC / ECG）
2. 通过 `input_current` 参数注入单位冲激（绕过 `input_linear`，确保每个神经元接收相同的 1.0 脉冲）
3. 记录 T=100 步的慢状态轨迹 `x_t` 和膜电位 `u⁰`
4. 差异纯粹来自每个神经元内在的谱参数 (α, ρ, ω)

## 数据集
- PS-MNIST（2 层，仅分析 Layer 0）
- GSC（1 层）
- ECG / QTDB（1 层）

## 输出
- `impulse_response_gallery.png` — 每个任务采样 8 个神经元的冲激响应线图（x_real 主线 + x_osc1/x_osc2 虚线），按 α 排序
- `frequency_response.png` — |FFT(x_real)| 频域图，颜色编码 α 值，展示低通/带通多样性
- `lif_comparison.png` — 每任务选慢/中/快 3 个神经元，SPRiF vs 等效 LIF exp(-t/τ) 对比 + 振荡幅度

## 解读指南
- **α 大（~0.9+）** → 快速衰减，短时记忆，类似 LIF 小 τ
- **α 小（~0.1-）** → 缓慢衰减，长时记忆
- **ω 大** → 振荡频率高，对周期性输入敏感
- **|FFT| 不单调衰减** → 带通滤波行为，LIF 不具备
- **SPRiF vs LIF 差异大** → 谱参数化产生了 LIF 无法表达的时域结构
