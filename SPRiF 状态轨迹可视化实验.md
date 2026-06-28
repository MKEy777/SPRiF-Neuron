## 实验：SPRiF 状态轨迹可视化实验

**Reset-Protected Spectral Trajectory Visualization**

------

# 1. 实验输入怎么做

每个样本只包含两个阶段：

$$
\mathrm{Cue} \rightarrow \mathrm{Delay\ with\ perturbation\ probes}
$$

不做分类，不做 match / mismatch，不需要 probe 阶段（与 delayed match-to-sample 等范式区分）。

------

## 1.1 时间长度

设置：

$$
\Delta t = 1\mathrm{ms}
$$

$$
T_{\mathrm{cue}} = 100\mathrm{ms}
$$

$$
T_{\mathrm{delay}} = 800\mathrm{ms}
$$

总长度：

$$
T = 900\mathrm{ms}
$$

------

## 1.2 输入通道

设置 32 个输入通道：

| 通道类型        | 数量 | 用途                         |
| --------------- | ---- | ---------------------------- |
| phase channels  | 20   | 编码 cue 的相位              |
| probe channels  | 10   | delay 阶段注入受控扰动电流 |
| marker channels | 2    | 标记 cue / delay             |

------

# 2. Cue 怎么生成

每个样本随机采样一个初始相位：

$$
\phi \sim \mathcal{U}(0,2\pi)
$$

随机采样一个频率：

$$
\omega \in
\left{
\frac{2\pi}{50},
\frac{2\pi}{100},
\frac{2\pi}{200}
\right}
$$

第 $i$ 个 phase channel 的 preferred phase 为：

$$
\varphi_i = \frac{2\pi i}{20}
$$

cue 阶段，也就是 $t=0$ 到 $100\mathrm{ms}$，第 $i$ 个通道的 Poisson firing rate 设置为：

$$
r_i(t)=r_0+r_1\cos(\omega t+\phi-\varphi_i)
$$

建议：

$$
r_0=30\mathrm{Hz}
$$

$$
r_1=25\mathrm{Hz}
$$

delay 阶段 phase channels 全部关闭：

$$
r_i(t)=0
$$

------

# 3. Controlled perturbation probes 怎么生成

在 delay 阶段插入**受控扰动探针（controlled perturbation probes）**。

设计逻辑类比电生理实验中的**电流钳协议（current-clamp protocol）**：实验者向神经元注入短暂去极化电流，在预定时刻诱发 spike，然后观察 spike 前后膜电位与内部状态的变化。本实验对 SPRiF 做同样的事——向 fast state 注入受控去极化电流，在已知时刻诱发 spike，从而干净地对比 spike 前后 slow state 与 fast state 的轨迹行为。

为了可视化清楚，建议先用固定 probe 位置：

$$
t_{\mathrm{probe}}
\in
{180,300,420,540,660,780}\mathrm{ms}
$$

注意这里的时间是整个 trial 中的绝对时间。

每个 probe 持续：

$$
T_{\mathrm{probe}}=10\mathrm{ms}
$$

probe channels 在 probe window 内以高频率发放：

$$
r_{\mathrm{probe}}=100\mathrm{Hz}
$$

probe window 外 probe channels 关闭。

同时给 SPRiF hidden layer 的 fast state 注入统一扰动电流：

$$
I_t \leftarrow I_t + A_{\mathrm{probe}} \cdot p_t
$$

其中：

$$
p_t=
\begin{cases}
1, & t \in \mathrm{probe\ window}\
0, & \mathrm{otherwise}
\end{cases}
$$

$A_{\mathrm{probe}}$ 调到能让 hidden neurons 在 probe window 内明显产生 spikes 即可。

------

# 4. 训练目标怎么做

这个实验不做分类。

让 SPRiF 在 cue 之后继续生成一个二维相位轨迹：

# $$ \mathbf{y}_t

\begin{bmatrix}
\cos(\phi+\omega t)\
\sin(\phi+\omega t)
\end{bmatrix}
$$

只在 delay 阶段计算损失。

也就是说，cue 阶段只负责把相位写入模型；delay 阶段没有相位输入，模型需要靠内部状态继续维持旋转轨迹。

------

# 5. 网络怎么搭

使用一个很小的 SPRiF 网络即可：

$$
\mathrm{Input}
\rightarrow
\mathrm{SPRiF\ Hidden\ Layer}
\rightarrow
\mathrm{Linear\ Readout}
\rightarrow
\hat{\mathbf{y}}_t
$$

建议设置：

| 项目                 | 设置                             |
| -------------------- | -------------------------------- |
| input dimension      | 32                               |
| SPRiF hidden neurons | 64                               |
| output dimension     | 2                                |
| recurrence           | 不启用                           |
| readout input        | hidden slow state $\mathbf{x}_t$ |
| readout output       | $\hat{\cos}_t,\hat{\sin}_t$      |

readout 可以写成：

# $$ \hat{\mathbf{y}}_t

W_{\mathrm{out}}
\mathrm{concat}
(\mathbf{x}*{1,t},...,\mathbf{x}*{N,t})
+
\mathbf{b}_{\mathrm{out}}
$$

### LIF 对照网络

为提供对照基线，同时训练一个参数匹配的 LIF 网络：

| 项目                 | 设置                             |
| -------------------- | -------------------------------- |
| input dimension      | 32                               |
| LIF hidden neurons   | 64                               |
| output dimension     | 2                                |
| recurrence           | 不启用                           |
| readout input        | hidden membrane potential $v_t$  |
| readout output       | $\hat{\cos}_t,\hat{\sin}_t$      |

LIF 网络与 SPRiF 共享完全相同的输入、输出维度和训练设置。关键差异在于：
- SPRiF readout 从慢状态 $\mathbf{x}_t$ 读取；LIF readout 从膜电位 $v_t$ 读取
- SPRiF 的 spike reset 只影响快状态；LIF 的 spike reset 直接影响膜电位（即唯一的记忆载体）

**对照预期**：LIF 在相同任务上也会尝试维持相位轨迹，但其膜电位在 spike 处被 reset，导致轨迹出现可见的断裂。这个对比直接展示 SPRiF 慢状态的结构性优势。

------

# 6. Loss 怎么写

只在 delay 阶段计算 MSE：

# $$ \mathcal{L}_{\mathrm{traj}}

\frac{1}{T_{\mathrm{delay}}}
\sum_{t\in\mathrm{delay}}
\left|
\hat{\mathbf{y}}_t-\mathbf{y}_t
\right|_2^2
$$

可以加一个很小的 firing-rate regularization，避免完全不放电：

# $$ \mathcal{L}

\mathcal{L}*{\mathrm{traj}}
+
\beta \mathcal{L}*{\mathrm{FR}}
$$

建议：

$$
\beta=10^{-4}
$$

------

# 7. 训练设置

| 项目               | 设置                 |
| ------------------ | -------------------- |
| train samples      | 10,000               |
| validation samples | 1,000                |
| batch size         | 64                   |
| optimizer          | AdamW                |
| learning rate      | $1\times10^{-3}$     |
| epochs             | 100                  |
| surrogate gradient | arctangent surrogate |
| gradient clipping  | 1.0                  |

训练时 perturbation probe 的位置可以随机 jitter：

$$
t_{\mathrm{probe}} \leftarrow t_{\mathrm{probe}}+\epsilon
$$

$$
\epsilon \sim \mathcal{U}(-20,20)\mathrm{ms}
$$

这样可视化时模型不会只记固定 probe 位置。

------

# 8. 训练后怎么做可视化样本

训练完成后，不需要统计准确率。

只挑 4 个固定样本做可视化。

设置相同频率：

$$
\omega=\frac{2\pi}{100}
$$

设置四个初始相位：

$$
\phi \in
\left{
0,
\frac{\pi}{2},
\pi,
\frac{3\pi}{2}
\right}
$$

perturbation probe 固定为：

$$
t_{\mathrm{probe}}
\in
{180,300,420,540,660,780}\mathrm{ms}
$$

对这 4 个样本，分别在 **SPRiF** 和 **LIF** 网络上做一次 forward pass，保存所有内部状态。

------

# 9. Forward 时保存什么

每个时间步保存：

| 变量                                               | 含义                   |
| -------------------------------------------------- | ---------------------- |
| input spikes $\mathbf{s}_t$                        | 输入 spike raster      |
| perturbation probe mask $p_t$                             | probe 位置       |
| slow state $\mathbf{x}_t$                          | SPRiF 慢状态           |
| fast pre-reset state $\mathbf{u}^{\mathrm{pre}}_t$ | reset 前快状态         |
| fast post-reset state $\mathbf{u}_t$               | reset 后快状态         |
| membrane $v_t$                                     | $u^{\mathrm{pre},0}_t$ |
| output spike $z_t$                                 | hidden spikes          |
| readout $\hat{\mathbf{y}}_t$                       | 输出轨迹               |
| target $\mathbf{y}_t$                              | 理论相位轨迹           |
| learned $\alpha,\rho,\omega,\lambda$               | SPRiF 参数             |

### LIF 对照网络保存变量

对 LIF 网络，每个时间步保存：

| 变量                          | 含义                     |
| ----------------------------- | ------------------------ |
| input spikes $\mathbf{s}_t$   | 输入 spike raster（同 SPRiF） |
| membrane potential $v_t$      | LIF 膜电位               |
| output spike $z_t$            | LIF hidden spikes        |
| readout $\hat{\mathbf{y}}_t$  | 输出轨迹                 |
| target $\mathbf{y}_t$         | 理论相位轨迹（同 SPRiF） |

**LIF 与 SPRiF 的对比维度**：

| 对比维度             | SPRiF                                    | LIF                                     |
| -------------------- | ---------------------------------------- | --------------------------------------- |
| 记忆载体             | 慢状态 $\mathbf{x}_t$（不被 reset）      | 膜电位 $v_t$（被 reset）                |
| spike 后记忆是否断裂 | 否（慢状态连续）                         | 是（膜电位被 reset）                    |
| 相位轨迹维持能力     | 预期：spike 后轨迹无断裂                 | 预期：spike 后轨迹可见跳变或相位漂移    |
| 可视化核心图         | Panel (b): $(x^1, x^2)$ 连续旋转         | 对照：$(v_t, \cdots)$ 轨迹在 spike 处断裂 |
| 可视化核心图         | Panel (c): $(u^0, u^1)$ 投影式 reset     | 对照：$v_t$ 标量 reset（归零或减量）    |

------

# 10. 图怎么画

## 10.1 AAAI 主文 Figure 布局

所有可视化整合为 **一个 5-panel Figure**，讲述完整的机制故事：从「spike 发生了」到「慢状态仍然连续」到「这就是投影式 reset」到「这个机制确实在解决问题」。

```
Figure X: SPRiF Functional State Decomposition Under Controlled Perturbation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
┌──────────────────────────────────────────────────────────────┐
│  (a) Task schematic (horizontal strip, ~1/5 page height)      │
│  [Cue 100ms] → [Delay 800ms ▾▾▾▾▾▾ probes]                  │
│  Readout target: (cos(φ+ωt), sin(φ+ωt))                       │
├────────────────────────────┬─────────────────────────────────┤
│  (b) Slow state portrait   │  (c) Fast state projective      │
│  (x¹, x²) phase plane      │  reset (u⁰, u¹) phase plane     │
│  Color gradient = time     │  ● = pre-spike position         │
│  ● = spike events          │  ▼ = post-spike position        │
│  "Continuous rotation      │  Arrows along [1, λⱼ]           │
│   through spike events"    │  "Directional, learnable reset"  │
│  ★ THE MONEY SHOT          │  ★ NOVELTY EVIDENCE             │
├────────────────────────────┴─────────────────────────────────┤
│  (d) Time-domain contrast around one spike (SPRiF vs LIF)     │
│  Top: SPRiF x¹(t) — smooth across spike                      │
│  Mid: SPRiF v(t) — crosses threshold, reset                  │
│  Bot: LIF v(t) — crosses threshold, reset → trajectory break  │
│  Vertical dashed line at spike time                           │
│  "SPRiF slow state continuous; membrane resets.              │
│   LIF: single state — memory = readout = reset target"        │
├──────────────────────────────────────────────────────────────┤
│  (e) Output verification (SPRiF vs LIF)                       │
│  Left: ŷ_cos(t) vs y_cos(t) in delay                         │
│  Right: (ŷ_cos, ŷ_sin) 2D circle vs target circle             │
│  "SPRiF maintains phase; LIF trajectory drifts"               │
└──────────────────────────────────────────────────────────────┘
```

------

## 10.2 Panel (a)：任务示意图

**保留程度**：极小示意图，占据 panel (a) 约 1/5 页高。

画一条简洁 timeline：

```text
0 ms        100 ms                                      900 ms
|---- Cue ----|---------------- Delay ----------------------|
              ▼     ▼     ▼     ▼     ▼     ▼
           probes at t = 180, 300, 420, 540, 660, 780 ms
```

标注：cue 阶段有相位输入（phase channels 编码 $\phi$）；delay 阶段无相位输入，模型须靠内部状态维持旋转；probe 时刻注入受控扰动电流。

**目的**：让读者 1 秒内理解实验范式，为 (b)-(e) 提供语境。

------

## 10.3 Panel (b)：慢状态相平面 — $(x_t^1, x_t^2)$【⭐ 核心视觉】

**保留程度**：主文核心 panel。对应原 Panel F。

画二维轨迹：

$$
(x_t^1,x_t^2),\quad t \in [100, 900]\mathrm{ms}
$$

- **横轴**：$x_t^1$，**纵轴**：$x_t^2$
- **颜色**：从 delay 起始（浅色）到 delay 结束（深色），连续 color gradient 编码时间
- **标记**：在 6 个 probe 时间点（即 spike 发生时刻）打实心圆点 ●
- **叠加**：可选画 LIF 的膜电位二维轨迹作为对照（同一坐标系，虚线/灰色），展示 LIF 轨迹在 spike 处断裂

**你要看到**：
- SPRiF 慢状态形成平滑的椭圆/圆形旋转轨迹
- 所有 probe 时刻的标记点**落在连续轨迹上**，无突变、无断裂
- （对照）LIF 膜电位轨迹在同一位置出现可见的不连续

**这张图是整篇论文最重要的可视化**，是「spike never resets memory」的直接视觉证据。建议放在 (b) 位置，占全图最大面积。

------

## 10.4 Panel (c)：快状态投影式 reset — $(u_t^0, u_t^1)$【⭐ 创新点证据】

**保留程度**：主文核心 panel。对应原 Panel G。

画快状态二维轨迹：

$$
(u_t^0,u_t^1)
$$

- 在 6 个 probe 诱导的 spike 时刻：
  - **实心圆 ●** = spike 前位置 $\mathbf{u}^{\mathrm{pre}}_t$
  - **三角 ▼** = spike 后位置 $\mathbf{u}_t$（reset 后）
  - **箭头** 连接 ● → ▼，方向沿 learnable reset vector $[1, \lambda_j]^T$

- reset 方向标注：

$$
\mathbf{u}_t = \mathbf{u}^{\mathrm{pre}}_t - \theta \begin{bmatrix} 1 \\ \lambda_j \end{bmatrix}
$$

- 可选：用不同颜色/箭头长度编码不同 neuron 的 $\lambda_j$ 值，展示 learnable 的多样性

**这张图是 SPRiF 最强创新点的视觉证据**（positioning 文档确认：「projective reset has zero known prior」）。必须清晰展示 reset 的**方向性**和**可学习性**。

------

## 10.5 Panel (d)：时间域对照 — SPRiF 慢状态 vs 膜电位 vs LIF 膜电位

**保留程度**：主文核心 panel。合并原 Panel D + E，并加入 LIF 对照。

选择同一个代表性 neuron，围绕一个 spike 事件（如 $t=420\mathrm{ms}$），取窗口 $[t_{\mathrm{spike}} - 30\mathrm{ms},\; t_{\mathrm{spike}} + 30\mathrm{ms}]$。

画三条时间曲线（上下对齐，共享时间轴）：

1. **SPRiF 慢状态分量**：$x_t^1$（或 $x_t^2$）
   - 展示曲线在 spike 时刻**平滑连续，无跳变**
2. **SPRiF 膜电位**：$v_t = u^{\mathrm{pre},0}_t$
   - 展示膜电位上升 → 超过 threshold → reset 跳变
   - spike 时刻标红点，threshold 画水平虚线
3. **LIF 膜电位**：$v_t^{\mathrm{LIF}}$
   - 展示膜电位上升 → 超过 threshold → 标量 reset
   - **关键对照**：LIF 的 reset 直接清零了唯一的记忆变量，而 SPRiF 的 reset 只影响快状态

spike 时刻画竖虚线贯穿三组曲线。

**这张图回答审稿人的核心怀疑**：「你怎么证明 spike 确实发生了，而慢状态确实没受影响？」——同一时间轴上的三层对照，无可辩驳。

------

## 10.6 Panel (e)：输出轨迹验证 — SPRiF vs LIF

**保留程度**：主文保留（较小）。对应原 Panel I。

画两类输出曲线：

**时间曲线**（上半）：
- $\hat{\cos}_t$ vs $\cos(\phi+\omega t)$（SPRiF 预测 vs 目标）
- 可选叠加 LIF 的 $\hat{\cos}_t$ 作为对照（虚线/灰色）
- 只画 delay 区间 $[100, 900]\mathrm{ms}$

**二维圆轨迹**（下半）：
- $(\hat{\cos}_t, \hat{\sin}_t)$ 在 2D 平面上的轨迹（SPRiF）
- 叠画目标圆 $(\cos, \sin)$
- 可选叠加 LIF 的轨迹作为对照

**这张图验证**：SPRiF 的慢状态连续不是无意义的——它确实让模型成功维持了相位轨迹。LIF 对照展示膜电位 reset 导致轨迹漂移/畸变。

------

## 10.7 LIF 对照可视化的核心要求

LIF 不是降级消融，而是**同一任务的另一种机制实现**。对照要展示的是**结构性差异**，而非性能高低：

| 要展示的对比 | SPRiF | LIF |
|-------------|-------|-----|
| 记忆载体在 spike 处的行为 | $\mathbf{x}_t$ 连续 | $v_t$ 跳变 |
| 轨迹形态 | 平滑旋转 | 在 spike 处可见断裂或相位跳变 |
| Reset 方式 | 投影式（方向性，在 2D 快状态中） | 标量式（膜电位减去常数值） |
| 输出质量 | 圆周轨迹完整 | 可能出现畸变、漂移或频率偏移 |

LIF 的失败是**结构性的、可预期的**——不是因为 LIF 不好，而是因为 LIF 没有独立的记忆状态。这正是 SPRiF 要解决的 gap。

------

## 10.8 移至附录的 Panels

以下 panel 不进入 AAAI 主文 7 页正文，但可作为附录/补充材料：

| Panel | 内容 | 移入附录的理由 |
|-------|------|---------------|
| **A** | 输入时间结构图 | 已被 Panel (a) 的简洁版替代；完整版细节过多 |
| **B** | 输入 spike raster | 原始数据展示，无科学洞察；审稿人不关心 phase channel 的 Poisson 发放形态 |
| **C** | Hidden spike raster | spike 时刻已标记在 Panel (b)(c)(d) 上；单独 raster 是信息重复 |
| **H** | Probe 局部放大 | 过于微观；Panel (d) 已覆盖同一时间窗口的核心信息。适合 rebuttal 或 PPT 演讲 |
| **J** | Learned spectral parameters 直方图 | 属于 Claim C6 的参数分析，应放在另一个独立的 Parameter Analysis Figure 中（见实验计划 Analysis 1），不放在轨迹图里 |

### 附录 Panel 的使用场景

- **B, C**：如果审稿人问「输入到底长什么样」「是不是所有 neuron 都同时 spike」，可调出附录图回答。rebuttal 时有用
- **H**：适合做 oral presentation 的幻灯片——放大一个 probe 窗口，逐帧展示「spike → u 跳变 → x 连续」
- **J**：放在主文 Parameter Analysis 部分（另一个 Figure），展示跨任务/跨层的 $\alpha, \rho, \omega$ 分布

------

## 10.9 多样本展示策略

主文 Figure 用 **1 个样本**（如 $\phi=0$）展示完整 5-panel 布局。

其余 3 个相位（$\phi=\pi/2, \pi, 3\pi/2$）的结果放在附录中，用简化的 2-panel 格式（仅 (b) 相平面 + (e) 输出轨迹），证明结果不是 cherry-picking。

------

## 10.10 频域增强（可选，如果空间允许）

可在 Panel (d) 下方叠加一个小频谱图：对 $x_t^1$ 在 spike 前后做短时傅里叶变换（STFT），展示 spike 时刻频谱成分保持稳定。进一步从频域证明「spike 不破坏慢状态的频率结构」。

如果空间不够，这个分析可以放入附录或完全省略。

