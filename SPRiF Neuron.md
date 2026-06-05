SPRiF Neuron：谱域投影重置脉冲神经元

## 1. 文档目的

本文档用于客观介绍 SPRiF 神经元的状态结构、更新流程、参数含义和当前实现形式。  

本文档暂不对该神经元的应用方向作预设，也不将其定义为某一类任务的专用模型。 SPRiF 可以被描述为一种对脉冲神经元内部动力学进行结构化设计的神经元单元。 它的核心特征是：神经元内部同时包含慢状态和快状态，慢状态负责连续的内部动态演化，快状态负责膜电位读出、spike 生成以及 spike 后的 reset。  

当前实验中，该神经元已经被用于多个时序任务，包括 GSC、QTDB、SHD、S-MNIST 和 PS-MNIST。 这些任务只作为实验覆盖范围记录，不在方法介绍部分用于提前限定 SPRiF 的适用方向或性能结论。  

## 2. LIF 神经元中的状态耦合问题



经典 LIF 神经元通常使用一个膜电位变量 $v_t$ 表示神经元内部状态。 其离散形式可以写为：  

$$v_t = \alpha v_{t-1} + I_t$$

其中，$v_t$ 是当前膜电位，$\alpha$ 是泄漏系数，$I_t$ 是当前输入电流。  

当膜电位超过阈值 $\theta$ 时，神经元产生 spike：  

$$z_t = H(v_t - \theta)$$

其中，$H(\cdot)$ 是阶跃函数。  

spike 发生后，膜电位通常会被重置。 例如，常见的软重置形式为：  

$$v_t \leftarrow v_t - z_t\theta$$

这种结构的优点是简单直接，但它也使得同一个变量 $v_t$ 同时承担三种作用：第一，保存过去输入的历史信息；第二，判断当前是否产生 spike；第三，在 spike 发生后接受 reset 操作。  

因此，在标准 LIF 中，记忆、放电和重置是耦合在同一个膜电位变量上的。  

从时间展开的角度看，LIF 的膜电位可以写为：  

$$v_t = \sum_{k=0}^{t}\alpha^k I_{t-k}$$

这说明标准 LIF 的历史记忆主要由单一指数衰减链表示。 该机制适合描述简单的泄漏积分过程，但对于更复杂的时间动态，例如多时间尺度变化、周期性变化或振荡型动态，其表达形式相对受限。  

SPRiF 的设计不是直接否定 LIF，而是将长期状态保存、放电判断和 spike 后重置这几个功能拆分到不同的内部状态中，使神经元的状态演化过程更加清晰。  

## 3. SPRiF 的总体结构



SPRiF 的全称是 Spectral Projective Reset Integrate-and-Fire Neuron，中文可以称为谱域投影重置积分发放神经元。  

SPRiF 将神经元内部状态分为两类：慢状态 $\mathbf{x}_t$ 和快状态 $\mathbf{u}_t$。 其中，慢状态 $\mathbf{x}_t$ 用于描述相对连续的内部动态，快状态 $\mathbf{u}_t$ 用于形成膜电位、生成 spike，并在 spike 发生后执行 reset。  

整体流程可以描述为：输入首先形成输入电流，输入电流驱动慢状态更新；更新后的慢状态再影响快状态；快状态的第一维读出为膜电位；膜电位超过阈值时产生 spike；spike 发生后，只对快状态执行投影式 reset，而慢状态继续保留为下一时刻的内部状态。  

因此，SPRiF 的单步更新包含五个部分：  

1. 输入电流计算；  
2. 慢状态更新；  
3. 快状态预更新；  
4. 膜电位读出与 spike 生成；  
5. 快状态 reset。  

为了避免符号混乱，本文统一使用 $\mathbf{u}^{\mathrm{pre}}_t$ 表示 reset 前的快状态，使用 $\mathbf{u}_t$ 表示 reset 后、真正保存到下一步的快状态。  

## 4. 输入电流

设当前时间步的输入为 $\mathbf{s}_t$。 对于一个神经元层，输入首先通过线性映射得到输入电流：  

$$\mathbf{I}_t = W_{\mathrm{in}}\mathbf{s}_t + \mathbf{b}$$

其中，$W_{\mathrm{in}}$ 是输入权重，$\mathbf{b}$ 是偏置项。  

如果该层启用递归连接，还可以加入上一时刻的输出 spike：  

$$\mathbf{I}_t = W_{\mathrm{in}}\mathbf{s}_t + W_{\mathrm{rec}}\mathbf{z}_{t-1} + \mathbf{b}$$

其中，$W_{\mathrm{rec}}$ 是递归权重，$\mathbf{z}_{t-1}$ 是上一时刻的 spike 输出。  

递归连接不是 SPRiF 神经元定义本身必须包含的部分，而是网络层实现时可以选择是否启用的结构。 因此，在介绍 SPRiF 神经元本身时，可以将输入电流统一记为 $\mathbf{I}_t$。  

对于单个神经元 $j$，其输入电流记为 $I_{j,t}$。 为了简化公式，下文在介绍单神经元动力学时省略神经元下标 $j$，直接写作 $I_t$。  

## 5. 慢状态：谱域状态更新



在当前实现中，每个 SPRiF 神经元的慢状态 $\mathbf{x}_t$ 是三维状态，可以写为：  

$$\mathbf{x}_t = \begin{bmatrix} x_t^{0} \\ x_t^{1} \\ x_t^{2} \end{bmatrix}$$

其中，$x_t^{0}$ 是实衰减分量，$x_t^{1}$ 和 $x_t^{2}$ 构成二维阻尼旋转分量。  

### 5.1 实衰减分量

慢状态第一维的更新为：  

$$x_t^{0} = \alpha x_{t-1}^{0} + (1-\alpha)I_t$$

其中，$\alpha$ 是实衰减系数，并满足 $0<\alpha<1$。  

这个分量可以理解为一个指数滑动状态。 它保留上一时刻状态的一部分，同时写入当前输入电流的一部分。  

### 5.2 阻尼旋转分量

慢状态第二维和第三维构成一个二维阻尼旋转系统：  

$$x_t^{1} = \rho \left( \cos\omega \cdot x_{t-1}^{1} - \sin\omega \cdot x_{t-1}^{2} \right) + (1-\rho)I_t$$

$$x_t^{2} = \rho \left( \sin\omega \cdot x_{t-1}^{1} + \cos\omega \cdot x_{t-1}^{2} \right)$$

其中，$\rho$ 是阻尼系数，$\omega$ 是旋转频率，并满足 $0<\rho<1$ 和 $0<\omega<\pi$。  

这两个分量可以看作一个带阻尼的二维旋转状态。 $\rho$ 控制状态幅度随时间衰减的速度，$\omega$ 控制状态在二维平面中的旋转速度。  

### 5.3 矩阵形式

慢状态更新也可以写成矩阵形式：  

$$\mathbf{x}_t = A\mathbf{x}_{t-1} + B I_t$$

其中：  

$$A = \begin{bmatrix} \alpha & 0 & 0 \\ 0 & \rho\cos\omega & -\rho\sin\omega \\ 0 & \rho\sin\omega & \rho\cos\omega \end{bmatrix}$$

输入项为：  

$$B I_t = \begin{bmatrix} (1-\alpha)I_t \\ (1-\rho)I_t \\ 0 \end{bmatrix}$$

因此，SPRiF 的慢状态同时包含实衰减模态和二维阻尼旋转模态。  

需要注意的是，慢状态 $\mathbf{x}_t$ 是当前时间步慢状态更新后的结果。 它不是 reset 前快状态，也不直接作为膜电位使用。 膜电位由快状态的第一维读出。  

## 6. 慢状态参数化

为了保证慢状态更新处于稳定范围内，SPRiF 对关键参数进行约束。 当前实现中，$\alpha$、$\rho$ 和 $\omega$ 都由可学习的 raw 参数经过 sigmoid 映射得到：  

$$\alpha = \sigma(\alpha_{\mathrm{raw}})$$

$$\rho = \sigma(\rho_{\mathrm{raw}})$$

$$\omega = \pi \cdot \sigma(\omega_{\mathrm{raw}})$$

这样可以保证 $\alpha$ 和 $\rho$ 位于 $(0,1)$ 区间，$\omega$ 位于 $(0,\pi)$ 区间。  

这种参数化的目的不是让状态转移矩阵完全自由，而是让慢状态具有受约束的谱结构。 也就是说，SPRiF 并不是任意使用一个三维 RNN 状态，而是在状态更新中显式保留实衰减和阻尼旋转两种结构。  

## 7. 快状态：放电相关状态

SPRiF 的快状态 $\mathbf{u}_t$ 是二维状态。 为了区分 reset 前后的快状态，本文使用 $\mathbf{u}^{\mathrm{pre}}_t$ 表示 reset 前的快状态，使用 $\mathbf{u}_t$ 表示 reset 后保存下来的快状态。  

当前时间步中，快状态预更新由上一时刻快状态 $\mathbf{u}_{t-1}$ 和当前慢状态 $\mathbf{x}_t$ 共同决定：  

$$\mathbf{u}^{\mathrm{pre}}_t = F(\mathbf{u}_{t-1}) + G\mathbf{x}_t$$

其中，$F(\mathbf{u}_{t-1})$ 表示快状态自身的短时演化，$G\mathbf{x}_t$ 表示慢状态到快状态的投影。  

在二维快状态下，可以写为：  

$$u^{\mathrm{pre},0}_t = \eta_0 u^{0}_{t-1} + \kappa u^{1}_{t-1} + (G\mathbf{x}_t)^0$$

$$u^{\mathrm{pre},1}_t = \eta_1 u^{1}_{t-1} + (G\mathbf{x}_t)^1$$

其中，$u^{\mathrm{pre},0}_t$ 和 $u^{\mathrm{pre},1}_t$ 分别表示 reset 前快状态的第一维和第二维；$u^{0}_{t-1}$ 和 $u^{1}_{t-1}$ 分别表示上一时刻 reset 后保存的快状态第一维和第二维。  

参数 $\eta_0$ 和 $\eta_1$ 是快状态泄漏系数，$\kappa$ 是快状态内部耦合系数，$G$ 是慢状态到快状态的投影矩阵。  

这里需要特别区分两个状态：  

- $\mathbf{u}^{\mathrm{pre}}_t$ 是当前时刻 reset 前的快状态；  
- $\mathbf{u}_t$ 是当前时刻 spike reset 后保存的快状态。  

后续膜电位读出和 spike 判断使用的是 $\mathbf{u}^{\mathrm{pre}}_t$，而下一时刻递推使用的是 reset 后的 $\mathbf{u}_t$。  

## 8. 膜电位读出与 spike 生成

在当前实现中，膜电位由 reset 前快状态的第一维读出：  

$$v_t = u^{\mathrm{pre},0}_t$$

然后将膜电位与阈值 $\theta$ 比较，得到 spike：  

$$z_t = H(v_t - \theta)$$

其中，$H(\cdot)$ 是阶跃函数。  

由于阶跃函数不可导，训练时通常使用 surrogate gradient 近似其反向传播梯度。 也就是说，前向传播中仍然生成二值 spike，但反向传播时使用一个平滑近似函数提供梯度。  

因此，SPRiF 仍然保留脉冲神经元的基本形式：膜电位超过阈值时产生 spike。 区别在于，膜电位不是单独的唯一内部状态，而是由快状态 $\mathbf{u}^{\mathrm{pre}}_t$ 的第一维读出。  

## 9. 投影式 Reset

spike 发生后，SPRiF 不直接 reset 慢状态 $\mathbf{x}_t$，而是对快状态执行 reset。  

reset 前快状态为 $\mathbf{u}^{\mathrm{pre}}_t$，reset 后快状态为 $\mathbf{u}_t$。 当前实现中的 reset 形式为：  

$$\mathbf{u}_t = \mathbf{u}^{\mathrm{pre}}_t - z_t \mathbf{r}_j \theta$$

其中，$\mathbf{r}_j$ 是第 $j$ 个神经元的 reset 方向：  

$$\mathbf{r}_j = \begin{bmatrix} 1 \\ \lambda_j \end{bmatrix}$$

这里，$\lambda_j$ 是可学习参数。 它控制该神经元在快状态第二维上的 reset 强度。  

将 reset 公式展开，可以得到：  

$$u^{0}_t = u^{\mathrm{pre},0}_t - z_t\theta$$

$$u^{1}_t = u^{\mathrm{pre},1}_t - z_t\lambda_j\theta$$

如果当前没有 spike，即 $z_t=0$，则有：  

$$\mathbf{u}_t = \mathbf{u}^{\mathrm{pre}}_t$$

如果当前发生 spike，即 $z_t=1$，则快状态沿方向 $\mathbf{r}_j$ 发生一次状态修正。  

这里的“投影式 reset”指的是，reset 不是只在一个标量膜电位上独立发生，也不是把所有内部状态清零，而是在二维快状态空间中沿着方向 $[1,\lambda_j]^T$ 执行一次定向修正。  

需要注意的是，当前实现中的 reset 幅度使用阈值 $\theta$。 因此，本文档不再引入额外的 $q_t$ 或 $\delta$ 记号，避免与当前代码实现不一致。  

## 10. 慢状态与快状态的时间顺序



为了避免符号混乱，SPRiF 在时间步 $t$ 的更新顺序可以明确写成下面几步。  

第一步，根据当前输入和上一时刻 spike 计算输入电流 $I_t$。  

第二步，根据上一时刻慢状态 $\mathbf{x}_{t-1}$ 和当前输入电流 $I_t$ 更新慢状态：  

$$\mathbf{x}_t = \mathrm{SlowFlow}(\mathbf{x}_{t-1}, I_t)$$

第三步，根据上一时刻 reset 后快状态 $\mathbf{u}_{t-1}$ 和当前慢状态 $\mathbf{x}_t$ 计算 reset 前快状态：  

$$\mathbf{u}^{\mathrm{pre}}_t = \mathrm{FastFlow}(\mathbf{u}_{t-1}, \mathbf{x}_t)$$

第四步，从 reset 前快状态第一维读出膜电位：  

$$v_t = u^{\mathrm{pre},0}_t$$

第五步，根据膜电位和阈值产生 spike：  

$$z_t = H(v_t - \theta)$$

第六步，根据 spike 对快状态执行 reset：  

$$\mathbf{u}_t = \mathbf{u}^{\mathrm{pre}}_t - z_t\mathbf{r}_j\theta$$

第七步，将 $\mathbf{x}_t$、$\mathbf{u}_t$ 和 $z_t$ 保存为下一时间步的状态。  

这个顺序说明，慢状态 $\mathbf{x}_t$ 在当前时间步先被更新，然后参与快状态预更新；spike 生成后，reset 只改变快状态，不回头修改当前慢状态。  

## 11. 与 LIF 的结构对比



标准 LIF 可以概括为：输入电流更新膜电位，膜电位用于 spike 判断，spike 发生后同一个膜电位变量被 reset。  

SPRiF 可以概括为：输入电流先更新慢状态，慢状态再影响快状态，快状态第一维读出膜电位，spike 发生后只对快状态执行 reset。  

两者的主要结构差异如下：  

| **方面**         | **LIF**             | **SPRiF**                                           |
| ---------------- | ------------------- | --------------------------------------------------- |
| 内部状态         | 单一膜电位 $v_t$    | 慢状态 $\mathbf{x}_t$ 与快状态 $\mathbf{u}_t$       |
| 记忆形式         | 单一指数衰减        | 实衰减分量与阻尼旋转分量                            |
| 膜电位来源       | 直接由 $v_t$ 表示   | 由 reset 前快状态第一维 $u^{\mathrm{pre},0}_t$ 读出 |
| spike 判断       | $z_t=H(v_t-\theta)$ | $z_t=H(u^{\mathrm{pre},0}_t-\theta)$                |
| reset 对象       | 膜电位 $v_t$        | 快状态 $\mathbf{u}_t$                               |
| 慢状态是否 reset | 不适用              | 不直接 reset                                        |
| reset 形式       | 标量 reset          | 快状态空间中的方向性 reset                          |

该对比只描述结构差异，不预设性能结论。  

## 12. 当前实现中的主要参数



当前实现中的 SPRiF 神经元层包含以下主要参数。  

### 12.1 输入映射参数



输入映射参数包括 $W_{\mathrm{in}}$ 和可选偏置 $\mathbf{b}$。 它们用于将当前输入 $\mathbf{s}_t$ 转换为输入电流 $\mathbf{I}_t$。  

如果启用递归连接，还包含递归权重 $W_{\mathrm{rec}}$，用于将上一时刻 spike $\mathbf{z}_{t-1}$ 加入当前输入电流。  

### 12.2 慢状态参数



慢状态相关参数包括 $\alpha_{\mathrm{raw}}$、$\rho_{\mathrm{raw}}$ 和 $\omega_{\mathrm{raw}}$。 它们经过 sigmoid 映射后得到 $\alpha$、$\rho$ 和 $\omega$。  

其中，$\alpha$ 控制实衰减分量，$\rho$ 控制阻尼旋转分量的衰减速度，$\omega$ 控制阻尼旋转分量的旋转频率。  

### 12.3 快状态参数



快状态相关参数包括 $\eta_{\mathrm{raw}}$、$\kappa$ 和 $G$。 其中，$\eta_{\mathrm{raw}}$ 经过 sigmoid 映射后得到快状态泄漏系数 $\eta_0$ 和 $\eta_1$；$\kappa$ 表示快状态第二维到第一维的内部耦合；$G$ 表示慢状态到快状态的投影矩阵。  

### 12.4 Reset 参数



reset 相关参数为 $\lambda_j$。 它与固定的第一维系数组成 reset 方向：  

$$\mathbf{r}_j = \begin{bmatrix} 1 \\ \lambda_j \end{bmatrix}$$

因此，spike 后快状态第一维执行标准阈值扣除，第二维根据 $\lambda_j$ 的大小执行同步修正。  



# introduction and method



# 1 引言

# 1 引言

脉冲神经网络通过离散 spike 序列在时间维度上传递和处理信息。与连续激活神经网络不同，SNN 的计算不仅由层间权重决定，也高度依赖神经元内部的状态演化方式：输入如何被累积，状态何时触发 spike，以及 spike 发生后状态如何被重置。对于语音、事件序列、生理信号和长序列输入等时序任务而言，神经元内部动态直接影响网络能够保留何种历史信息，以及如何在离散 spike 事件之后继续建模后续序列。

经典泄漏积分发放神经元，即 LIF 神经元，因其简洁性和可训练性而被广泛采用。在标准离散 LIF 中，单一膜电位变量同时承担三个角色：它积分过去输入，用于判断当前是否产生 spike，并在 spike 之后接受 reset。这种膜电位中心式设计构成了许多 SNN 方法的基础，但也引入了一个结构性耦合：时间记忆、放电判定和 spike 后重置被绑定在同一个状态变量上。从时间展开的角度看，标准 LIF 的历史记忆主要表现为单一指数衰减 trace，因此其内部动态更接近简单泄漏积分过程。对于需要多时间尺度、周期性或振荡型响应的序列，这种单一膜电位状态可能并不是最合适的状态组织方式。

这一观察引出本文关注的问题：**spike 发生后，是否必须重置承载时间记忆的同一个状态变量？** 在标准 LIF 中，答案隐含地是肯定的，因为膜电位既是记忆变量，也是放电变量和 reset 变量。然而，从时序建模角度看，spike 更像是当前状态达到阈值后的离散事件；它需要修正放电相关状态，却不一定意味着所有内部记忆都应被同步削弱或清除。因此，一个自然的设计方向是：在保留阈值触发二值 spike 的同时，将持续时间记忆与 spike 后 reset 分离。

本文提出 **SPRiF**，即 **Spectral Projective Reset Integrate-and-Fire Neuron**，一种面向时序脉冲神经网络的结构化神经元动力学。SPRiF 的核心思想是重新分配标准 LIF 中耦合在膜电位上的功能：慢状态用于保存连续时间动态，快状态用于膜电位读出、spike 生成以及 spike 后 reset。换言之，SPRiF 将“负责记忆的状态”和“负责放电与重置的状态”显式区分开来。这样，spike 事件可以作用于放电相关的快状态，而不必直接重置承载时间记忆的慢状态。

为了使慢状态具有更结构化的时间响应，SPRiF 使用受约束的谱域动态来表示神经元内部记忆。该慢状态包含实衰减模态和阻尼旋转模态，从而在标准单指数膜电位 trace 之外提供更丰富的时间滤波形式。与此同时，SPRiF 在快状态空间中执行 spike-triggered projective reset，使 reset 成为作用于放电状态的定向更新，而不是对全部内部状态的统一清零。通过这种慢/快状态分工，SPRiF 在保留阈值触发二值 spike 输出的同时，为 SNN 提供了一种介于标准 LIF 与完全自由循环状态之间的结构化神经元设计。

SPRiF 并不是将 SNN 替换为普通循环神经网络。首先，SPRiF 的输出仍然是由阈值机制产生的二值 spike 序列。其次，SPRiF 的慢状态转移不是任意可学习的隐藏状态转移，而是由受约束的衰减和旋转模态构成。第三，递归连接只是网络层实现中的可选结构，而不是 SPRiF 神经元定义本身的必要组成部分。因此，SPRiF 更适合被理解为一种结构化 spiking neuron dynamics，而不是一个通用 RNN cell。

我们将 SPRiF 作为一种神经元级别的动力学设计进行研究，而不是作为某一特定任务的专用模型。实验部分在 GSC、QTDB、SHD、S-MNIST 和 PS-MNIST 等多类时序任务上评估 SPRiF，覆盖语音、生理信号、事件序列和长序列输入等不同场景。为了检验其结构设计是否真正有效，我们进一步使用统一网络骨架下的神经元替换实验、参数量匹配对比、核心机制消融以及学习到的动力学可视化，分析谱域慢状态、慢/快状态解耦和投影式 reset 对模型行为的贡献。

本文的主要贡献如下：

1. 提出 **SPRiF**，一种结构化脉冲神经元，将标准 LIF 中耦合在单一膜电位上的时间记忆、spike 生成和 reset 功能拆分为慢状态与快状态。

2. 引入受约束的谱域慢状态，用实衰减模态和阻尼旋转模态共同描述神经元内部的连续时间动态，为 SNN 提供比单一指数膜电位 trace 更丰富的结构化时间响应。

3. 提出投影式快状态 reset，使 spike 后的 reset 沿可学习方向作用于放电相关快状态，而不直接重置慢状态。

4. 在多类时序任务上评估 SPRiF，并通过机制消融和动力学可视化分析谱域慢状态、慢/快状态分离和投影式 reset 的作用。



# 2 方法

## 2.1 问题设定：膜电位中心式 LIF 动态

为了说明 SPRiF 的设计动机，我们首先回顾标准 LIF 神经元的离散更新形式。设 $I_t$ 表示时间步 $t$ 的输入电流，$v_t$ 表示膜电位，$\alpha$ 表示泄漏系数，则标准 LIF 的积分过程可以写为

$$
v_t=\alpha v_{t-1}+I_t .
$$

当膜电位超过阈值 $\theta$ 时，神经元产生 spike：

$$
z_t=H(v_t-\theta),
$$

其中 $H(\cdot)$ 是阶跃函数。spike 发生后，膜电位通常被重置。例如在软重置形式下，

$$
v_t \leftarrow v_t-z_t\theta .
$$

这个过程简洁有效，但也意味着同一个膜电位变量 $v_t$ 同时承担三个功能：历史输入的记忆、当前 spike 的判定、以及 spike 后 reset 的目标。若不考虑 reset 的非线性影响，LIF 的膜电位可以展开为

$$
v_t=\sum_{k=0}^{t}\alpha^k I_{t-k},
$$

即标准 LIF 的时间记忆主要由单一指数衰减链表示。该结构适合描述泄漏积分过程，但对更复杂的时间动态，例如多时间尺度变化、周期性结构或振荡型动态，其表达形式较为有限。

SPRiF 的目标不是否定 LIF，而是在保留阈值发放和二值 spike 输出的前提下，重新组织神经元内部状态。具体来说，我们希望将持续时间记忆与放电相关状态区分开，使 spike 后 reset 不必直接作用于承载慢时间动态的状态变量。

---

## 2.2 SPRiF 总体结构

SPRiF 将每个神经元的内部状态拆分为两部分：慢状态 $\mathbf{x}_t$ 和快状态 $\mathbf{u}_t$。慢状态用于保存连续的内部时间动态；快状态用于形成膜电位、产生 spike，并在 spike 后接受 reset。为了区分 reset 前后的快状态，我们使用 $\mathbf{u}^{\mathrm{pre}}_t$ 表示当前时间步 reset 前的快状态，使用 $\mathbf{u}_t$ 表示 reset 后保存到下一时间步的快状态。

SPRiF 在时间步 $t$ 的整体流程如下：

$$
\mathbf{s}_t
\rightarrow
\mathbf{I}_t
\rightarrow
\mathbf{x}_t
\rightarrow
\mathbf{u}^{\mathrm{pre}}_t
\rightarrow
v_t
\rightarrow
z_t
\rightarrow
\mathbf{u}_t .
$$

其中，$\mathbf{s}_t$ 是输入，$\mathbf{I}_t$ 是输入电流，$\mathbf{x}_t$ 是慢状态，$\mathbf{u}^{\mathrm{pre}}_t$ 是 reset 前快状态，$v_t$ 是膜电位，$z_t$ 是 spike 输出，$\mathbf{u}_t$ 是 reset 后快状态。

SPRiF 的单步更新可以概括为七个步骤：

1. 根据当前输入和可选的上一时刻 spike 计算输入电流；
2. 根据输入电流更新慢状态；
3. 将慢状态投影到快状态，并计算 reset 前快状态；
4. 从 reset 前快状态第一维读出膜电位；
5. 根据膜电位和阈值产生 spike；
6. 若产生 spike，则对快状态执行投影式 reset；
7. 保存慢状态、reset 后快状态和 spike，作为下一时间步的递推状态。

这一更新顺序说明：慢状态在当前时间步先被更新，并参与快状态预更新；spike 发生后，reset 只改变快状态，不回头修改当前慢状态。

---

## 2.3 输入电流

设当前时间步的输入为 $\mathbf{s}_t$。对于一个 SPRiF 神经元层，输入首先经过线性映射得到输入电流：

$$
\mathbf{I}_t=W_{\mathrm{in}}\mathbf{s}_t+\mathbf{b},
$$

其中 $W_{\mathrm{in}}$ 是输入权重，$\mathbf{b}$ 是偏置项。

如果该层启用递归连接，则还可以加入上一时刻的输出 spike：

$$
\mathbf{I}_t=W_{\mathrm{in}}\mathbf{s}_t+W_{\mathrm{rec}}\mathbf{z}_{t-1}+\mathbf{b}.
$$

这里 $W_{\mathrm{rec}}$ 是递归权重，$\mathbf{z}_{t-1}$ 是上一时刻的 spike 输出。需要注意的是，递归连接不是 SPRiF 神经元定义的必要部分，而是层级实现中的可选结构。因此，在介绍单个神经元的动力学时，我们将输入电流统一记为 $I_t$。

---

## 2.4 谱域慢状态

SPRiF 的慢状态 $\mathbf{x}_t$ 是三维状态：

$$
\mathbf{x}_t=
\begin{bmatrix}
x_t^0\\
x_t^1\\
x_t^2
\end{bmatrix}.
$$

其中，$x_t^0$ 是实衰减分量，$(x_t^1,x_t^2)$ 构成二维阻尼旋转分量。慢状态的作用是保存相对连续的内部时间动态，而不是直接作为膜电位使用。

### 2.4.1 实衰减分量

慢状态第一维按照如下方式更新：

$$
x_t^0=\alpha x_{t-1}^0+(1-\alpha)I_t,
$$

其中 $\alpha$ 是实衰减系数，满足 $0<\alpha<1$。该分量可以理解为一个指数滑动状态：它保留上一时刻状态的一部分，同时写入当前输入电流的一部分。

### 2.4.2 阻尼旋转分量

慢状态第二维和第三维构成二维阻尼旋转系统：

$$
x_t^1
=
\rho
\left(
\cos\omega \cdot x_{t-1}^1
-
\sin\omega \cdot x_{t-1}^2
\right)
+
(1-\rho)I_t,
$$

$$
x_t^2
=
\rho
\left(
\sin\omega \cdot x_{t-1}^1
+
\cos\omega \cdot x_{t-1}^2
\right).
$$

其中，$\rho$ 是阻尼系数，$\omega$ 是旋转频率，并满足 $0<\rho<1$ 和 $0<\omega<\pi$。$\rho$ 控制二维旋转状态随时间衰减的速度，$\omega$ 控制状态在二维平面中的旋转速度。

### 2.4.3 矩阵形式

慢状态更新也可以写成矩阵形式：

$$
\mathbf{x}_t=A\mathbf{x}_{t-1}+BI_t,
$$

其中

$$
A=
\begin{bmatrix}
\alpha & 0 & 0\\
0 & \rho\cos\omega & -\rho\sin\omega\\
0 & \rho\sin\omega & \rho\cos\omega
\end{bmatrix},
$$

$$
BI_t=
\begin{bmatrix}
(1-\alpha)I_t\\
(1-\rho)I_t\\
0
\end{bmatrix}.
$$

该形式表明，SPRiF 的慢状态并不是任意隐藏状态，而是由一个实衰减模态和一个二维阻尼旋转模态组成的受约束谱结构。

---

## 2.5 慢状态参数化

为了将慢状态控制在稳定范围内，SPRiF 对关键参数进行约束。具体地，$\alpha,\rho,\omega$ 由可学习 raw 参数经过如下映射得到：

$$
\alpha=\sigma(\alpha_{\mathrm{raw}}),
$$

$$
\rho=\sigma(\rho_{\mathrm{raw}}),
$$

$$
\omega=\pi\sigma(\omega_{\mathrm{raw}}),
$$

其中 $\sigma(\cdot)$ 是 sigmoid 函数。因此，$\alpha$ 和 $\rho$ 被限制在 $(0,1)$ 区间，$\omega$ 被限制在 $(0,\pi)$ 区间。

这一参数化有两个作用。第一，它避免慢状态转移矩阵成为完全自由的三维状态转移。第二，它使每个参数具有明确动力学含义：$\alpha$ 控制实衰减时间尺度，$\rho$ 控制阻尼旋转模态的衰减速度，$\omega$ 控制旋转频率。

---

## 2.6 快状态与膜电位读出

在 SPRiF 中，快状态 $\mathbf{u}_t$ 是二维状态。与慢状态不同，快状态直接参与膜电位读出、spike 生成以及 spike 后 reset。当前时间步中，reset 前快状态由上一时刻 reset 后快状态 $\mathbf{u}_{t-1}$ 和当前慢状态 $\mathbf{x}_t$ 共同决定：

$$
\mathbf{u}^{\mathrm{pre}}_t
=
F(\mathbf{u}_{t-1})+G\mathbf{x}_t,
$$

其中 $F(\mathbf{u}_{t-1})$ 表示快状态自身的短时演化，$G\mathbf{x}_t$ 表示慢状态到快状态的投影。

在二维快状态下，可以展开为：

$$
u_t^{\mathrm{pre},0}
=
\eta_0 u_{t-1}^{0}
+
\kappa u_{t-1}^{1}
+
(G\mathbf{x}_t)^0,
$$

$$
u_t^{\mathrm{pre},1}
=
\eta_1 u_{t-1}^{1}
+
(G\mathbf{x}_t)^1.
$$

其中，$\eta_0$ 和 $\eta_1$ 是快状态泄漏系数，$\kappa$ 是快状态第二维到第一维的内部耦合系数，$G$ 是慢状态到快状态的投影矩阵。

膜电位由 reset 前快状态的第一维读出：

$$
v_t=u_t^{\mathrm{pre},0}.
$$

这点与标准 LIF 不同。在 LIF 中，膜电位本身就是唯一内部状态；在 SPRiF 中，膜电位只是快状态的一维读出，且该读出发生在 reset 之前。

---

## 2.7 Spike 生成与 surrogate gradient 训练

给定膜电位 $v_t$ 和阈值 $\theta$，SPRiF 的 spike 生成为

$$
z_t=H(v_t-\theta).
$$

其中 $z_t\in\{0,1\}$ 表示当前时间步是否产生 spike。由于阶跃函数 $H(\cdot)$ 不可导，训练时使用 surrogate gradient 近似其反向传播梯度。需要强调的是，surrogate gradient 只用于反向传播；前向传播中，SPRiF 仍然产生二值 spike。

因此，SPRiF 保留了脉冲神经元的基本输出形式，而不是将输出替换为连续激活值。

---

## 2.8 投影式快状态 Reset

标准 LIF 通常在 spike 后直接对膜电位执行标量 reset。SPRiF 则在二维快状态空间中执行投影式 reset。设 reset 前快状态为 $\mathbf{u}^{\mathrm{pre}}_t$，reset 后快状态为 $\mathbf{u}_t$，则

$$
\mathbf{u}_t
=
\mathbf{u}^{\mathrm{pre}}_t
-
z_t\mathbf{r}_j\theta,
$$

其中 $\mathbf{r}_j$ 是第 $j$ 个神经元的 reset 方向：

$$
\mathbf{r}_j=
\begin{bmatrix}
1\\
\lambda_j
\end{bmatrix}.
$$

这里 $\lambda_j$ 是可学习参数，用于控制 spike 后快状态第二维的 reset 强度。将上式展开可得：

$$
u_t^0
=
u_t^{\mathrm{pre},0}
-
z_t\theta,
$$

$$
u_t^1
=
u_t^{\mathrm{pre},1}
-
z_t\lambda_j\theta.
$$

当 $z_t=0$ 时，

$$
\mathbf{u}_t=\mathbf{u}^{\mathrm{pre}}_t.
$$

当 $z_t=1$ 时，快状态沿方向 $[1,\lambda_j]^T$ 被修正。第一维执行标准阈值扣除，第二维根据可学习参数 $\lambda_j$ 进行同步修正。该 reset 机制不直接作用于慢状态 $\mathbf{x}_t$。因此，SPRiF 将 spike 后的放电相关修正限制在快状态空间中，而让慢状态继续保存当前已经更新出的内部时间动态。

---

## 2.9 单步更新算法

为了清晰描述 SPRiF 的时间递推过程，下面给出单个时间步的算法形式。

**算法 1：SPRiF 神经元单步更新**

输入：当前输入 $\mathbf{s}_t$，上一时刻慢状态 $\mathbf{x}_{t-1}$，上一时刻 reset 后快状态 $\mathbf{u}_{t-1}$，上一时刻 spike $\mathbf{z}_{t-1}$。

输出：当前 spike $\mathbf{z}_t$，当前慢状态 $\mathbf{x}_t$，当前 reset 后快状态 $\mathbf{u}_t$。

1. 计算输入电流：

$$
\mathbf{I}_t=W_{\mathrm{in}}\mathbf{s}_t+\mathbf{b},
$$

若启用递归连接，则

$$
\mathbf{I}_t=W_{\mathrm{in}}\mathbf{s}_t+W_{\mathrm{rec}}\mathbf{z}_{t-1}+\mathbf{b}.
$$

2. 对每个神经元，根据谱域慢状态更新公式计算：

$$
\mathbf{x}_t=\mathrm{SlowFlow}(\mathbf{x}_{t-1}, I_t).
$$

3. 根据上一时刻快状态和当前慢状态计算 reset 前快状态：

$$
\mathbf{u}^{\mathrm{pre}}_t
=
\mathrm{FastFlow}(\mathbf{u}_{t-1},\mathbf{x}_t).
$$

4. 从快状态第一维读出膜电位：

$$
v_t=u_t^{\mathrm{pre},0}.
$$

5. 产生 spike：

$$
z_t=H(v_t-\theta).
$$

6. 执行投影式 reset：

$$
\mathbf{u}_t
=
\mathbf{u}^{\mathrm{pre}}_t
-
z_t
\begin{bmatrix}
1\\
\lambda_j
\end{bmatrix}
\theta.
$$

7. 保存 $\mathbf{x}_t$、$\mathbf{u}_t$ 和 $z_t$，用于下一时间步。

该算法突出了 SPRiF 的关键顺序：先更新慢状态，再计算快状态与 spike，最后只对快状态执行 reset。

---

## 2.10 SPRiF 层与网络读出

在层级实现中，多个 SPRiF 神经元并行组成一个 SPRiF 神经元层。给定长度为 $T$ 的输入序列，层在每个时间步重复执行上述单步更新，并输出对应的 spike 序列：

$$
\mathbf{Z}
=
[\mathbf{z}_1,\mathbf{z}_2,\ldots,\mathbf{z}_T].
$$

对于多层网络，可以将上一层输出的 spike 序列作为下一层输入。最后一层输出的 spike 序列可以进一步接线性 readout，用于时间步级别或序列级别预测。

以当前 ECG 任务代码为例，`SPRiFECGModel` 使用多个 `SPRiFNeuronLayer` 堆叠形成时序特征提取器，最后接一个线性 readout 产生分类 logits。该模型根据 `mode` 参数决定是否启用递归连接：当 `mode="srnn"` 时，SPRiF 层使用递归输入；否则可以作为非递归 SNN 层使用。

需要强调的是，ECG 代码只是 SPRiF 的一个任务实现。SPRiF 本身是神经元层级的动力学设计，不应被限定为 ECG 专用模型。

---

## 2.11 与标准 LIF 的结构对比

SPRiF 与标准 LIF 的区别可以概括如下。

| 方面             | 标准 LIF            | SPRiF                                               |
| ---------------- | ------------------- | --------------------------------------------------- |
| 内部状态         | 单一膜电位 $v_t$    | 慢状态 $\mathbf{x}_t$ 与快状态 $\mathbf{u}_t$       |
| 时间记忆形式     | 单一指数衰减链      | 实衰减模态 + 阻尼旋转模态                           |
| 膜电位来源       | 直接由 $v_t$ 表示   | 由 reset 前快状态第一维 $u_t^{\mathrm{pre},0}$ 读出 |
| spike 生成       | $z_t=H(v_t-\theta)$ | $z_t=H(u_t^{\mathrm{pre},0}-\theta)$                |
| reset 对象       | 膜电位 $v_t$        | 快状态 $\mathbf{u}_t$                               |
| 慢状态是否 reset | 不适用              | 不直接 reset                                        |
| reset 形式       | 标量 reset          | 快状态空间中的方向性 reset                          |

这个对比只描述结构差异，并不预设性能结论。SPRiF 的设计假设是：将持续时间记忆和 spike 后 reset 分离，并用受约束谱结构增强慢状态，可以为时序 SNN 提供更清晰的内部动力学组织方式。该假设需要通过主实验、机制消融、参数量匹配和动态分析共同验证。

---

## 2.12 机制分析：有效时间响应

SPRiF 的谱域慢状态还可以从有效时间响应的角度理解。对于标准 LIF，若忽略 reset 非线性，其时间响应主要由

$$
\alpha^k
$$

形式的单一指数核决定。相比之下，SPRiF 的慢状态包含实衰减和阻尼旋转两类响应。对于实衰减分量，输入脉冲经过 $k$ 个时间步后的贡献近似为

$$
(1-\alpha)\alpha^k.
$$

对于二维阻尼旋转分量，其响应包含

$$
(1-\rho)\rho^k\cos(k\omega),
$$

以及

$$
(1-\rho)\rho^k\sin(k\omega).
$$

因此，SPRiF 的慢状态可以表示指数衰减项与阻尼振荡项的组合。这为模型提供了一种结构化的时间滤波基，使其在保留 spike 输出的同时，具备比标准 LIF 单一指数 trace 更丰富的内部时间动态。这里的分析来自慢状态公式本身；至于这些动态在不同任务中是否带来性能提升，需要通过消融实验和 learned parameter visualization 进一步验证。

---

## 2.13 实现与可分析参数

SPRiF 的每个参数都具有明确动力学含义。慢状态参数 $\alpha,\rho,\omega$ 分别控制实衰减、阻尼强度和旋转频率；快状态参数 $\eta_0,\eta_1$ 控制快状态泄漏，$\kappa$ 控制快状态内部耦合，$G$ 控制慢状态到快状态的投影；reset 参数 $\lambda_j$ 控制 spike 后快状态第二维的修正强度。

这些参数可用于论文中的机制可视化。例如，可以比较不同任务上学习到的 $\alpha,\rho,\omega$ 分布，分析模型是否使用不同时间尺度和频率结构；也可以对齐 spike 时刻，观察 spike 前后慢状态 $\mathbf{x}_t$ 与快状态 $\mathbf{u}_t$ 的轨迹变化，从而验证 reset 主要作用于快状态，而慢状态保持连续演化。

---

## 2.14 方法小结

综上，SPRiF 可以被视为一种结构化的脉冲神经元单元。它保留了积分发放神经元的基本形式：输入电流驱动状态更新，膜电位超过阈值后产生二值 spike，并在 spike 后执行 reset。但与标准 LIF 不同，SPRiF 不再让单一膜电位同时承担记忆、放电和 reset 三个功能，而是通过慢状态与快状态的分离实现状态功能重分配。

慢状态 $\mathbf{x}_t$ 以受约束谱结构保存连续时间动态，包含实衰减和阻尼旋转模态；快状态 $\mathbf{u}_t$ 接收慢状态投影，产生膜电位读出，并在 spike 后沿可学习方向执行投影式 reset。这个设计的核心并不是简单增加状态维度，而是将时间记忆、spike 判断和 reset 操作分别放置在具有明确动力学角色的状态中。最终，SPRiF 为时序 SNN 提供了一种介于标准 LIF 与完全自由循环状态之间的结构化神经元设计。
