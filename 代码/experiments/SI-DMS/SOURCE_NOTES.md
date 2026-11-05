# Source and adaptation notes

本实现独立保存在本目录，不修改外部代码。

## ASRNN

参考本工作区 `ASRNN/GSC/SRNN_layers/spike_neuron.py`：

- `alpha = exp(-dt/tau_m)` 与 `rho = exp(-dt/tau_adp)`；
- 自适应状态 `b = rho*b + (1-rho)*spike`；
- 动态阈值 `threshold + beta*b`；
- 非放电 leaky-integrator readout 的实验架构习惯。

本目录的 `ASRNNCell` 将这些更新适配到统一 SI-DMS 接口。它是外部基线，不是 SPRiF 消融。

## BRF

参考 `ianjoshi/brf-neurons-reproduction` 中的 `snn/modules/rf.py`、`lif.py`、`alif.py` 以及模型包装方式：

- LIF 使用指数衰减积分与 scalar soft reset；
- BRF 使用二维 resonant 状态 `(u,v)`、refractory state `q` 和稳定性约束；
- hidden spikes 接非放电输出积分器。

公开仓库版本检查于提交 `f5ad3e6d31362d0bb834beea1ad62d54a216dd49`。本实现只移植神经元更新语义并统一输入、readout、优化器和干预协议，从而减少基线之间的实验管线差异。

## SPRiF

- `sprif_full`：3-D slow state 与 2-D fast state 分离，spike 后对 fast state 施加 `[1, λ]` learned projective reset，slow state 不被重置。
- `sprif_lambda0`：与 full 具有相同 slow/fast 状态和动力学，仅固定 reset 为 `[1, 0]`，且不含可训练 λ。
- `sprif_merged`：单一 3-D state，不再拥有独立 fast state，只对膜坐标进行 scalar reset。

这三项定义分别对应本工作区已有 SPRiF full、ablation C (`lambda0`) 与 ablation B (`merged`) 的机制边界。
