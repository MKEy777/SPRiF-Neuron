# AAAI 27 Paper Decision and Reviews
## English Version and Chinese Translation

The AAAI-27 paper decision is Reject. This document presents the decision and all three reviews in English, followed by a Chinese translation.

## English Original

### Paper Decision
**By:** Program Chairs  
**Date:** 25 September 2026, 02:45 (modified: 25 September 2026, 18:31)  
**Recipients:** Program Chairs; Authors; Program Committee (Submitted); Area Chairs; Secondary Senior Program Committee; Senior Program Committee

**Decision: Reject**

#### Comment
This paper did not advance to Phase 2 review for the AAAI-27 conference. The paper was reviewed by at least two human reviewers and at least one senior member of the program committee in Phase 1. In addition, papers with at least one positive review were examined by one program chair or associate program chair to ensure the rejection was justified. This decision is final. The provided reviews contain more detail, which we hope will be beneficial to the progress of this research.

### Official Review by Program Committee wAme
**Review title:** Good paper, some doubts on the experiments  
**Date:** 31 August 2026, 21:21 (modified: 25 September 2026, 05:33)  
**Recipients:** Program Chairs; Area Chairs; Senior Program Committee; Program Committee wAme; Authors; Program Committee (Submitted)

#### Paper Summary
The work introduces SPRiF, a spiking network architecture that decouples the state in charge of spiking from the state in charge of temporal memory. By doing this, the model is allowed to retain information through spiking episodes. This is demonstrated by experimental results on sequence classification benchmarks and experiments on random spiking interventions. Ablations on the different components of SPRiF are performed, highlighting that decoupling is a significant feature of the model.

#### Strengths
1. The paper is well-presented. Exposition is clear.
1. The architecture represents a genuine novelty in the literature of spiking neural networks.
1. The model matches or outperforms most baselines on multiple datasets.
1. Additional ablation and experiments are well-motivated and represent valuable insight.

#### Weaknesses
Most concerns to me are in the experimental part.

1. Baseline results seem to be only reported, not reproduced; as a consequence, variance is missing.
1. The ablation study for the merged model could also be run at the same parameter count to ensure that performance degradation comes mostly from architectural features and not from fewer parameters, as noticed in the work.
1. SSC, a dataset on which most of the baselines were tested, is missing. This dataset is one of the most difficult in terms of accuracy percentage, unlike others that are nearly saturated, so it could be a valuable addition.

#### Justification of Recommendation
The paper seems a significant advance in spiking neural networks, thus I am leaning towards acceptance.

#### Specific Points of Feedback for Rebuttal
As noticed in the paper, the merged variant is influenced by lower parameter count. Have the authors tried to match the parameter count?

**Rating:** 6: Marginally above acceptance threshold  
**Confidence:** 2: The reviewer is willing to defend the evaluation, but it is quite likely that the reviewer did not understand central parts of the paper

### Official Review by Program Committee HQsu
**Review title:** Review of SPRiF  
**Date:** 17 August 2026, 20:40 (modified: 25 September 2026, 05:33)  
**Recipients:** Program Chairs; Area Chairs; Senior Program Committee; Program Committee HQsu; Authors; Program Committee (Submitted)

#### Paper Summary
This paper proposes SPRiF, a novel spiking neuron that decouples temporal memory from spike generation and reset. Each neuron maintains a slow spectral state for context storage and a fast discharge state for spike readout. Experimental results on five benchmark datasets show that SPRiF achieves competitive performance.

#### Strengths
1. The paper correctly identifies the fundamental limitation of LIF—coupling memory, firing, and reset in a single state—and addresses it through clean state separation.
1. The spectral construction is explicit, bounded via sigmoid constraints, and yields closed-form spike responses.
1. The experimental results demonstrate that SPRiF achieves competitive performance.

#### Weaknesses
1. Increased per-neuron state complexity. SPRiF's additional dynamic equations require the maintenance of extra state variables and the performance of extra matrix multiplication operations. This undermines the low-power characteristics of spiking neural networks, which stem from event-driven, purely additive operations.
1. The novelty is weak. Previous studies have explored the distinction between the long-term slow state and the short-term fast state of spiking neurons. This paper should discuss and compare those studies, thoroughly elaborating on the unique advantages of the proposed method.
1. Modest accuracy margins. Although experiments were conducted on five datasets, the results in Table 1 suggest that the proposed method is not significantly more effective than existing methods. In fact, its performance on GSC and SHD is inferior.
1. Limited initialization and training dynamics analysis. The paper introduces several learnable parameters but does not discuss how they are initialized, whether training is sensitive to initialization, or whether they saturate during optimization.

#### Justification of Recommendation
The idea that slow states maintain long-term memory while fast states generate immediate responses has already been explored in the field of SNNs and does not represent a unique contribution. Moreover, the complexity of the proposed method undermines the low-power advantage of SNNs' event-driven computation. The method also did not yield significant performance gains in the experiments.

#### Specific Points of Feedback for Rebuttal
The authors should provide further responses regarding the novelty, efficiency, and performance advantages of their method.

**Rating:** 3: Clear rejection  
**Confidence:** 5: The reviewer is absolutely certain that the evaluation is correct and very familiar with the relevant literature

### AI Review
**By:** AI Reviewer  
**Date:** 27 July 2026, 19:59 (modified: 25 September 2026, 05:33)  
**Recipients:** Program Chairs; AI Reviewer; Area Chairs; Senior Program Committee; Authors; Program Committee (Submitted)

**Paper title:** SPRiF: Spectral Projective Reset Integrate-and-Fire Neuron

#### Synopsis of the Paper
The paper introduces a spiking neuron that separates temporal memory from spike discharge and reset. A structured slow state combines decaying and rotational modes, while a fast state produces spikes and receives a learned directional reset. Experiments cover temporal classification, component ablations, forced-spike interventions, and learned temporal kernels.

#### Summary of Review
The formulation provides an explicit stable slow transition and precisely separates direct reset from the slow state while acknowledging later recurrent spike effects. The specific combination of per-neuron spectral memory, fast discharge, and learned reset direction is clearly specified, but the comparison with closely related resonant and state-space models is incomplete. The main validity issue is that the central ablations change effective state capacity, so they do not isolate state separation or rotation. SI-DMS provides substantial evidence under intervention-aware training, but not architecture-only robustness to unforeseen spike corruption. The benchmark results establish competitiveness, while the reported cross-paper rankings do not establish controlled superiority.

#### Strengths
- The central mechanism is mathematically auditable. Equations (6)–(13) specify the update order, constrain the slow spectrum, derive its impulse response, and distinguish direct reset from subsequent effects through $W_{\mathrm{rec}} z_t$. This makes the claimed form of memory/reset separation precise and experimentally testable.
- SI-DMS directly targets the motivating failure mode rather than relying only on classification accuracy. Table 3 compares paired clean and stressed replays, includes ASRNN with matched clean accuracy, and contrasts full SPRiF with $\lambda = 0$; the large separation at unseen intervention counts provides informative evidence under the tested stress protocol.
- The empirical analysis connects learned dynamics to the formal construction. Figures 3–4 examine reset directions, state trajectories, and impulse responses corresponding to Eq. (10), showing heterogeneous decay, rotational, and reset behavior rather than collapse to one common configuration.

#### Weaknesses
- Table 1 supports only the explicitly qualified claim of highest accuracy “among the compared methods,” not a broader strongest-reported conclusion. The table combines published results from different protocols without baseline uncertainty, while the cited D-RF model is omitted despite reporting 99.50, 98.20, and 96.20 on S-MNIST, PS-MNIST, and SHD, respectively, versus SPRiF’s 99.28, 95.86, and 91.52 (Zhang et al., 2025). Moreover, SPRiF’s S-MNIST margin is only 0.08 percentage points against its own standard deviation of 0.06, so the comparison establishes literature-level competitiveness rather than controlled superiority.
- The main ablations do not isolate the paper’s central mechanisms. In Table 2, the merged model reduces the dynamic state from five coordinates to three, while setting $\omega = 0$ makes the second oscillatory coordinate input-independent because Eq. (7) reduces it to $x_t^2 = \rho x_{t-1}^2$. The reported losses therefore conflate separation and rotation with effective input-driven capacity; equal-dimensional merged and non-rotational controls are needed for causal attribution.
- The novelty boundary against closely related published alternatives remains underdeveloped. SpikingSSMs is cited only in the generic statement that structured state-space dynamics process long dependencies, despite integrating structured temporal memory with spiking dynamics, while Rhythm-SNN is absent despite studying oscillatory temporal preservation and robustness across the same benchmark families (Shen et al., 2025; Yan et al., 2025). This matters because the contribution should be evaluated as SPRiF’s specific spectral-memory, fast-discharge, and directional-reset combination rather than the broader ideas of oscillatory memory or separate temporal processing stages.
- SI-DMS establishes extrapolation to larger intervention counts after training on the same intervention type, rather than intrinsic robustness to unexpected spikes. The SI-DMS section states that training samples $K \in \{0,1,2,4,8\}$, while evaluation extends to larger $K$, and each forced spike jointly triggers native reset and enters the recurrent pathway. Table 3 therefore does not establish architecture-only robustness without intervention training or isolate direct-reset protection from recurrent network effects.

#### Suggestions for Improvement
- Add an efficiency analysis that reflects dynamic-state cost rather than trainable parameters alone. Equation (6) maintains five state coordinates per neuron and applies the projection $G$, whereas Table 1 reports only parameter counts; state memory, operations per timestep, firing rates, and measured latency would clarify the computational consequences of the construction.
- Organize the experimental section around mechanism-level hypotheses stated before Table 2. For example, rotation should affect frequency-sensitive processing, reset direction should affect intervention robustness, and separation should affect retention during continued spiking; this would distinguish confirmatory evidence from descriptive diagnostics.
- Introduce the complete neuron through a compact table or schematic listing each state’s role, update timing, dimensionality, and learnable parameters before Eq. (6). This would reduce the need to reconstruct the functions of the second fast coordinate, $\kappa$, $G$, and $\lambda$ from several later subsections.

#### Additional Details and Minor Issues
- Equation (11) is an affine translation along $[1,\lambda]^{\mathsf T}$, not a projection in the standard linear-algebraic sense; “projective reset” should be defined precisely or renamed.
- Equations (2) and (6) use previous-step and same-step reset conventions, respectively; the timing correspondence should be stated.
- Figure 4 labels a mode as $\alpha = 1.00$, although Eq. (9) enforces $\alpha < 1$; additional precision would clarify the rounding.
- The main text should state whether $\theta$, $\kappa$, and $G$ are learned and how $\eta_0$ and $\eta_1$ are constrained.
- SI-DMS would be easier to interpret with a concise definition of its cue generation, target balance, delays, and readout before the intervention procedure.

#### References
- Shen, S., Wang, C., Huang, R., Zhong, Y., Guo, Q., Lu, Z., Zhang, J., & Leng, L. (2025). SpikingSSMs: Learning long sequences with sparse and parallel spiking state space models. Proceedings of the AAAI Conference on Artificial Intelligence, 39(19), 20380–20388.

- Yan, Y., Yang, Q., Wu, Y., Liu, H., Zhang, M., Li, H., Tan, K. C., & Wu, J. (2025). Efficient and robust temporal processing with neural oscillations modulated spiking neural networks. Nature Communications, 16, 8651.

- Zhang, D., Zhang, M., Wang, S., Wang, J., Wei, W., Ma, Z., Wang, G., Yang, Y., & Li, H. (2025). Dendritic resonate-and-fire neuron for effective and efficient long sequence modeling. Advances in Neural Information Processing Systems, 38.

---
## 中文译文

### 论文决定
**发布方：** 程序主席  
**时间：** 2026年9月25日 02:45（修改于2026年9月25日 18:31）  
**接收对象：** 程序主席；作者；程序委员会（已提交）；领域主席；第二级高级程序委员会；高级程序委员会

**决定：拒稿**

#### 评审决定说明
该论文未能进入 AAAI-27 会议的第二阶段评审。第一阶段至少由两位人工评审人和一位程序委员会高级成员评审。此外，对于至少收到一份正面评审意见的论文，程序主席或副程序主席也会进行审查，以确保拒稿决定合理。本决定为最终决定。所提供的评审意见包含更多细节，我们希望这些意见能有助于这项研究的后续推进。

### 程序委员会评审 wAme
**评审标题：** 论文不错，但对实验有些疑虑  
**时间：** 2026年8月31日 21:21（修改于2026年9月25日 05:33）  
**接收对象：** 程序主席；领域主席；高级程序委员会；程序委员会评审人 wAme；作者；程序委员会（已提交）

#### 论文摘要
本文提出了 SPRiF，一种脉冲网络架构，将负责产生脉冲的状态与负责保存时间记忆的状态分离。通过这种设计，模型能够在连续脉冲发放期间保留信息。作者在序列分类基准上的实验结果以及随机脉冲干预实验展示了这一点。论文还对 SPRiF 的不同组成部分进行了消融，结果表明状态解耦是模型的一项重要特征。

#### 优点
1. 论文呈现良好，阐述清晰。
1. 该架构在脉冲神经网络领域具有真正的新颖性。
1. 在多个数据集上，该模型达到或超过了大多数基线方法。
1. 补充的消融和实验动机充分，并提供了有价值的见解。

#### 缺点
我最主要的疑虑集中在实验部分。

1. 基线结果似乎只是引用报告值，并未复现，因此缺少方差数据。
1. 可以在相同参数量下对合并模型进行消融，以确认性能下降主要来自架构特征，而不是参数更少；论文中已经注意到了这一点。
1. 缺少 SSC 数据集上的实验，而大多数基线方法都在该数据集上进行了测试。与准确率已接近饱和的其他数据集不同，SSC 是准确率方面最具挑战性的数据集之一，因此加入该数据集可能很有价值。

#### 推荐意见说明
这篇论文似乎是脉冲神经网络领域的一项重要进展，因此我倾向于接收。

#### rebuttal 阶段的具体反馈
如论文中所述，合并变体受到参数量较低的影响。作者是否尝试过匹配参数量？

**评分：** 6：略高于接收门槛  
**信心：** 2：评审人愿意为该评价辩护，但评审人很可能没有理解论文的核心部分

### 程序委员会评审 HQsu
**评审标题：** 对 SPRiF 的评审  
**时间：** 2026年8月17日 20:40（修改于2026年9月25日 05:33）  
**接收对象：** 程序主席；领域主席；高级程序委员会；程序委员会评审人 HQsu；作者；程序委员会（已提交）

#### 论文摘要
本文提出了 SPRiF，一种新型脉冲神经元，将时间记忆与脉冲生成及复位解耦。每个神经元都维护一个用于存储上下文的慢速光谱状态，以及一个用于读取脉冲的快速放电状态。在五个基准数据集上的实验结果表明，SPRiF 达到了有竞争力的性能。

#### 优点
1. 论文正确指出了 LIF 的根本局限：将记忆、发放和复位耦合在同一个状态中；并通过清晰的状态分离来解决这一问题。
1. 光谱构造定义明确，通过 sigmoid 约束保持有界，并能得到脉冲响应的闭式表达。
1. 实验结果表明 SPRiF 具有有竞争力的性能。

#### 缺点
1. 单个神经元的状态复杂度增加。SPRiF 增加的动态方程需要维护额外的状态变量，并执行额外的矩阵乘法运算。这削弱了脉冲神经网络的低功耗特性，而这种低功耗特性源于事件驱动、纯加法的运算方式。
1. 新颖性较弱。此前研究已经探讨过脉冲神经元的长期慢状态与短期快状态之间的区别。本文应讨论并比较这些研究，充分阐明所提方法的独特优势。
1. 准确率优势有限。尽管在五个数据集上进行了实验，表1的结果表明，所提方法并未显著优于现有方法。事实上，它在 GSC 和 SHD 上的表现较差。
1. 对初始化和训练动态的分析有限。论文引入了若干可学习参数，但未讨论这些参数如何初始化、训练是否对初始化敏感，以及它们在优化过程中是否会饱和。

#### 推荐意见说明
慢状态保留长期记忆、快状态产生即时响应的思想已经在脉冲神经网络领域得到探索，并不构成独特贡献。此外，所提方法的复杂度削弱了脉冲神经网络利用事件驱动计算实现低功耗的优势。实验中，该方法也没有带来显著的性能提升。

#### rebuttal（答辩）阶段的具体反馈
作者应进一步回应该方法在新颖性、效率和性能优势方面的问题。

**评分：** 3：明确拒稿  
**信心：** 5：评审人确信该评价正确，并且对相关文献非常熟悉

### AI 评审
**评审方：** AI Reviewer  
**时间：** 2026年7月27日 19:59（修改于2026年9月25日 05:33）  
**接收对象：** 程序主席；AI Reviewer；领域主席；高级程序委员会；作者；程序委员会（已提交）

**论文标题：** SPRiF: Spectral Projective Reset Integrate-and-Fire Neuron（SPRiF：光谱投影复位积分发放神经元）

#### 论文概要
本文提出了一种脉冲神经元，将时间记忆与脉冲放电及复位分离。结构化的慢状态结合衰减模态和旋转模态，而快状态负责产生脉冲，并接受一个可学习的方向性复位。实验涵盖时间序列分类、组成部分消融、强制脉冲干预和学习得到的时间核。

#### 评审总结
该构造给出了显式且稳定的慢状态转移，并清楚地区分了直接复位与慢状态，同时指出后续仍会产生循环脉冲效应。。每个神经元的光谱记忆、快速放电和可学习方向性复位这一特定组合得到了清楚说明，但与密切相关的共振模型和状态空间模型的比较并不完整。主要的有效性问题在于，核心消融改变了有效状态容量，因此无法单独检验状态分离或旋转的作用。SI-DMS 在面向干预的训练设置下提供了大量证据，但没有证明模型仅凭架构本身就能抵御未预见的脉冲扰动。基准结果表明该方法具有竞争力，而跨论文排名并不能证明其在受控条件下优于其他方法。

#### 优点
- 核心机制在数学上可审查。公式（6）–（13）规定了更新顺序、约束慢状态的谱、推导其脉冲响应，并区分直接复位与后续影响，其中涉及 $W_{\mathrm{rec}} z_t$。这使论文所声称的记忆与复位分离形式明确且可通过实验检验。
- SI-DMS 直接针对论文的动机性失效模式，而不只是依赖分类准确率。表3比较了成对的干净回放和受压回放，纳入干净准确率匹配的 ASRNN，并将完整 SPRiF 与 $\lambda = 0$ 进行比较；在未见过的干预次数下出现的大幅差异，为所测试的压力协议提供了有信息量的证据。
- 实证分析将学到的动态与形式化构造联系起来。图3–4 检查了与公式（10）相对应的复位方向、状态轨迹和脉冲响应，显示出不同的衰减、旋转和复位行为，而非全部塌缩为同一种配置。

#### 缺点
- 表1只支持经过明确限定的“在所比较的方法中准确率最高”这一说法，不能支持更宽泛的“报告结果中最强”结论。该表汇总了不同实验协议下发表的结果，却未提供基线方差；与此同时，论文没有列出引用的 D-RF 模型，尽管该模型在 S-MNIST、PS-MNIST 和 SHD 上分别报告了 99.50、98.20 和 96.20，而 SPRiF 的相应结果为 99.28、95.86 和 91.52（Zhang 等，2025）。此外，SPRiF 在 S-MNIST 上的优势仅为 0.08 个百分点，而其自身标准差为 0.06。因此，该比较只能说明其在文献报告结果层面具有竞争力，不能证明其在受控条件下占优。
- 主要消融没有单独检验论文的核心机制。表2中的合并模型将动态状态从五个坐标减少到三个；而设置 $\omega = 0$ 会使第二个振荡坐标不再依赖输入，因为公式（7）会将其化简为 $x_t^2 = \rho x_{t-1}^2$。因此，报告的性能损失混合了状态分离、旋转和有效输入驱动容量的影响。要作出因果归因，还需要维度相同的合并模型和非旋转对照。
- 对密切相关的已发表方法，新颖性边界讨论不足。论文仅在“结构化状态空间动态可处理长依赖”这一一般性论述中引用了 SpikingSSMs，尽管该方法将结构化时间记忆与脉冲动态结合；论文也未提及 Rhythm-SNN，尽管后者研究了振荡式时间保持，并在相同的基准数据集系列上评估了鲁棒性（Shen 等，2025；Yan 等，2025）。这一点很重要，因为应当将 SPRiF 的贡献界定为光谱记忆、快速放电和方向性复位的特定组合，而不是更宽泛的振荡记忆或分离式时间处理阶段。
- SI-DMS 证明的是：在针对同一种干预类型进行训练后，模型可以外推到更大的干预次数；它并未证明模型能固有地抵御意外脉冲。SI-DMS 部分称训练样本采用 $K \in \{0,1,2,4,8\}$，评估则扩展至更大的 $K$；每个强制脉冲都会同时触发原生复位并进入循环通路。因此，表3既未证明在没有干预训练时模型仅凭架构就具有鲁棒性，也未将直接复位的保护作用与循环网络效应区分开来。

#### 改进建议
- 增加能够反映动态状态成本的效率分析，而不应只报告可训练参数。公式（6）为每个神经元维护五个状态坐标，并施加投影 $G$；而表1只报告了参数量。补充状态内存、每个时间步的运算量、发放率和实测延迟，将有助于说明该构造带来的计算代价。
- 在表2之前先提出机制层面的假设，并围绕这些假设组织实验部分。例如，旋转应影响频率敏感处理；复位方向应影响干预鲁棒性；状态分离应影响持续脉冲发放期间的信息保留。这样可以区分验证性证据与描述性诊断结果。
- 在公式（6）之前，用一张简明表格或示意图介绍完整神经元，列出每种状态的作用、更新时间、维度和可学习参数。这样可以避免读者从后续多个小节中自行推断第二个快状态坐标、$\kappa$、$G$ 和 $\lambda$ 各自的功能。

#### 补充细节与次要问题
- 公式（11）表示沿 $[1,\lambda]^{\mathsf T}$ 的仿射平移，按标准线性代数术语来说并非投影；应准确界定“投影式复位”，或考虑更名。
- 公式（2）和（6）分别采用前一时间步和当前时间步的复位约定；应说明两者在时间上的对应关系。
- 图4将一个模态标为 $\alpha = 1.00$，而公式（9）约束其为 $\alpha < 1$；更精确的表述可说明舍入方式。
- 正文应说明 $\theta$、$\kappa$ 和 $G$ 是否为可学习参数，以及 $\eta_0$ 和 $\eta_1$ 如何受到约束。
- 在介绍干预过程之前，简要说明 SI-DMS 的线索生成方式、目标平衡、延迟和读出方法，将有助于读者理解该任务。

#### 参考文献
- Shen, S., Wang, C., Huang, R., Zhong, Y., Guo, Q., Lu, Z., Zhang, J., & Leng, L. (2025). SpikingSSMs: Learning long sequences with sparse and parallel spiking state space models. Proceedings of the AAAI Conference on Artificial Intelligence, 39(19), 20380–20388.

- Yan, Y., Yang, Q., Wu, Y., Liu, H., Zhang, M., Li, H., Tan, K. C., & Wu, J. (2025). Efficient and robust temporal processing with neural oscillations modulated spiking neural networks. Nature Communications, 16, 8651.

- Zhang, D., Zhang, M., Wang, S., Wang, J., Wei, W., Ma, Z., Wang, G., Yang, Y., & Li, H. (2025). Dendritic resonate-and-fire neuron for effective and efficient long sequence modeling. Advances in Neural Information Processing Systems, 38.
