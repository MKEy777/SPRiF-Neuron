# Conference Review — SPRiF: Spectral Projective Reset Integrate-and-Fire Neuron

## 1. Report Metadata

- **Review date:** 2026-07-19
- **Target venue/year/track:** AAAI-27, Main Technical Track
- **Paper title:** SPRiF: Spectral Projective Reset Integrate-and-Fire Neuron
- **Input materials reviewed:** `SPRiF_AAAI2027.tex`、`SPRiF_AAAI2027_supp.tex`、对应的 8 页主稿 PDF 与 9 页补充 PDF、`sprif2027.bib`、编译日志；另只为合规核验检查了独立的 `ReproducibilityChecklist.tex/.pdf`。未把本地训练代码或结果文件作为本次科学结论的证据。
- **Search basis:** AAAI-27 官方 CFP、投稿规则与补充材料规则；公开安全关键词检索 two-compartment SNN、slow-memory/fast-spiking decomposition、state-space spiking neurons、reset mechanisms。没有把未公开正文粘贴到网络查询中。
- **Report file:** `ccfa-review-reports/2026-07-19-sprif-aaai2027-conference-review.md`
- **Reviewer mode:** Standard；完整科学审稿、多审稿人模拟与 AC/meta-review

## 2. Desk Rejection Assessment

- **Paper length — pass.** 主稿为 8 页；技术内容在第 7 页结束，第 8 页只有参考文献。AAAI-27 允许最多 7 页技术内容、总长最多 9 页，超过第 7 页的页面仅可用于参考文献与 reproducibility checklist，当前页分配符合规则。[AAAI-27 Main Technical Track CFP](https://aaai.org/conference/aaai/aaai-27/main-technical-track-call/)
- **Topic compatibility — pass.** 面向长时序处理的新型脉冲神经元，同时连接机器学习、神经形态计算与动力系统，符合 AAAI Main Technical Track 的广义 AI 范围。
- **Minimum quality — pass.** 主稿具备完整问题定义、更新方程、稳定性约束、五个基准、机制消融、干预实验、动力学分析和限制性措辞，达到可审稿门槛。
- **Policy/anonymity/compliance — uncertain / font failure.** 主稿和补充材料均匿名，未发现作者、单位、邮件、本地路径或外部匿名性泄漏；独立 reproducibility checklist 已填写。主稿 PDF 的 Figure 2 含未嵌入 `ArialMT`，补充 PDF 的 Figure S1 含未嵌入 `Arial-BoldMT`。AAAI 要求 trouble-free、高分辨率、US-Letter PDF 并使用合规字体；未嵌入字体可能触发上传检查、替换字形或显示异常，提交前必须修复。[AAAI-27 Submission Instructions](https://aaai.org/conference/aaai/aaai-27/submission-instructions/)
- **Prompt injection and hidden manipulation detection — pass.** 对主稿与补充源文件进行了 reviewer/LLM 指令、白色文字、零字号、隐藏接受指令等模式扫描，未发现可疑内容。
- **Ethics and reviewability — pass with minor uncertainty.** 使用公开标准数据集，不涉及新的人体实验、敏感数据或明显滥用场景；硬件与软件环境已披露。数据许可、环境成本和更完整的限制仍可补强，但当前不构成伦理拒稿点。

**Desk rejection risk:** medium，直至两处未嵌入字体被修复；修复后为 low。

**Reason:** 页数、匿名性、主题和最低质量通过，但 PDF 字体合规是可被自动检查捕获的实际风险。

**Can be fixed before review?** yes。

## 3. Paper Summary And Contribution Map

本文提出 SPRiF，将每个神经元的内部状态分成三维 slow spectral state 与二维 fast discharge state。慢状态由一个实衰减模态和一对阻尼旋转模态组成，参数化保证谱半径小于 1；快状态读取慢状态、产生二值 spike，并沿学习到的方向 `[1, lambda]` 接受 projective reset。其严格成立的结构性结论是：同一时间步内，local reset 对慢状态的直接增量为零，但 recurrent spike 仍可在下一时间步通过普通递归输入影响慢状态。实验包括五个时序分类基准、三个机制变体、SI-DMS 强制 spike 干预任务、reset-direction 统计以及学习到的 temporal kernel 分析。

- **Claimed problem:** 标准 LIF 用同一个膜变量承担历史积分、放电与 reset，导致通信事件可能扰动所保留的上下文。
- **Claimed gap:** 单纯增加时间常数、适应变量或振荡模态，并不等于把 memory 与 resettable discharge 分离。
- **Method/contribution map:**（1）per-neuron functional state decomposition；（2）稳定且可解释的实模态加旋转模态 slow state；（3）二维 fast state 与 learned projective reset；（4）SI-DMS 机制诊断。
- **Evidence package:** 五任务五种子 SPRiF 均值与标准差；跨论文基线表；三个任务上的三种机制变体；三种子 SI-DMS；状态轨迹、lambda 分布、相关性与 impulse-response 分析；补充材料中的 loss-landscape 和 feature-perturbation diagnostics。
- **Stated limitations:** 跨论文协议不匹配；merged 变体同时改变状态维度；kernel 对比仅覆盖两个任务；更大规模和神经形态硬件实验留待未来。

## 4. Search And Related-Work Basis

- **Queries used:** `two-compartment spiking neuron long-term memory`；`slow memory pathway fast spiking neural network`；`structured state space spiking neuron`；`reset mechanism spiking sequential modeling`；公开论文标题检索。
- **Sources searched:** AAAI proceedings、ICML 官方页面、OpenReview、Nature Machine Intelligence、arXiv 与 DBLP。
- **Closest works found:**
  - [TC-LIF](https://ojs.aaai.org/index.php/AAAI/article/download/29625/31061) 使用 dendritic/somatic two-compartment dynamics 学习长依赖；已在正文和表格中讨论。
  - [LSTM-LIF](https://arxiv.org/abs/2307.07231) 把 dendritic long-term memory 与 somatic discharge 区分；已在正文中讨论，但与 SPRiF 的功能分离差异仍只用一句话说明。
  - [DMP-SNN](https://www.nature.com/articles/s42256-026-01255-3) 明确采用 slow memory pathway 与 fast spiking pathway；正文已加入其 layer-shared 与 SPRiF per-neuron 的区别，也已加入 S-/PS-MNIST 表格，是最接近的原则级先行工作。
  - [CLIF](https://icml.cc/virtual/2024/poster/32664) 通过 complementary state/path 改善 temporal gradient，同时保持 binary output。BibTeX 中已有该条目，但正文 related work 未引用或解释其与 fast/slow state 路径的区别。
  - [SpikingSSMs](https://ojs.aaai.org/index.php/AAAI/article/download/34245/36400) 与 SiLIF 将结构化 state-space dynamics 引入 SNN；正文已概括，但缺少状态、reset、训练复杂度的直接对照。
- **Unverified related-work risks:** 未穷尽 2026 年 5 月以后所有非同行评审预印本；AAAI-27 规则也不要求作者掌握提交截止前两个月内发表的工作。
- **Source-quality screening status:** 使用官方会议/期刊页面和原始论文页面；未使用低质量聚合站点支撑新颖性结论。

## 5. Expected Review Outcome

- **Expected outcome:** borderline negative / weak reject；存在较高 Phase-1 淘汰风险
- **Main accept signal:** 具体神经元构造清楚、稳定、可解释，reset boundary 写得严谨；SI-DMS 比常规纯准确率表更接近机制验证，作者也主动限制了跨论文和因果措辞。
- **Main reject signal:** 主基准缺少同架构、同训练、同种子、同参数预算的重训基线；最核心的 state-separation 消融仍被总状态维度混淆；SI-DMS 在训练阶段已经见过较弱干预且测量的是 reset 与 recurrent propagation 的联合效应。
- **Confidence:** 4/5。完整主稿、补充材料、官方规则和针对性 related-work 搜索均可用；匿名代码包、原始逐种子结果和独立复现实验未纳入本次审查。

## 6. Strengths And Weaknesses

### Strengths

- 更新顺序、状态维度、参数约束和 reset 作用边界定义明确。Eqs. 6、12、13 足以支持“同一步 direct local reset 不直接编辑慢状态”这一窄结论。
- 慢状态的谱结构不是任意黑箱递归，而是一个实衰减模态和一个阻尼旋转对；稳定性、impulse response 与参数含义均可审计。
- 五个任务覆盖顺序图像、ECG、frame-based speech 与 event-based speech，说明模型并非只对一种编码有效。
- 主稿主动把跨论文数字称为 descriptive comparison，并明确 merged 变体的容量混淆、SI-DMS 的 recurrent-spike 语义以及两任务 kernel 对比的范围，科学措辞总体克制。
- SI-DMS 包含 matched batches、matched intervention masks、forced-hit audit、训练外 K 值与 intervention-fraction sweep，实验设计比一般“画状态轨迹”更有说服力。
- PDF 总体清晰、无裁切或重叠；主稿四张核心图和三张表均可读。

### Weakness 1: 主基准缺少同协议重训基线

- **Weakness:** Table 1 的主要证据是跨论文数字，虽然作者正确地将其限定为 descriptive context，但论文仍以这些数字承担“practical neuron design”和参数效率的主要外部有效性证据。
- **Evidence basis:** Main Table 1 与 Experiments 第 281–295 行；不同方法使用不同网络、预处理、训练与模型选择协议。只有 SI-DMS 对 LIF、ASRNN、BRF 做了统一宽度和统一干预控制。
- **Reviewer deduction:** 无法判断提升来自 neuron dynamics，还是来自任务特定的训练配置、网络宽度、读出或模型选择。S-MNIST 的领先幅度仅 0.08 个百分点，尤其不应被跨协议排序解释为方法优势。
- **Required fix:** 至少在 PS-MNIST、GSC、SHD 三个代表性任务上，以相同 architecture、optimizer、epochs、seeds、readout 与参数预算重训 LIF/ASRNN/BRF/TC-LIF 或最强可复现基线，报告 mean±std、逐种子配对差异和训练成本。

### Weakness 2: 核心 state-separation 结论没有被容量匹配地隔离

- **Weakness:** merged 变体同时把五维 slow+fast state 变为三维共享状态，因此改变了功能分工、状态容量、参数数量和优化几何。
- **Evidence basis:** Main lines 299–317；Supplement Table S2 明确承认 merged 不保持总状态维度。SI-DMS 中 merged 的 clean accuracy 只有 88.0%，显著低于 full 的 100.0%。
- **Reviewer deduction:** 当前结果表明“这个具体 merged 小模型更差”，但不能把差异唯一归因于 memory/discharge separation。该问题直接命中贡献 1，是当前最重要的机制证据缺口。
- **Required fix:** 增加 five-state capacity-matched merged control，保持总状态数、可训练参数、输入投影和 fast recurrence 尽可能一致，只改变 reset 是否直接进入 memory-carrying coordinates；并在相同种子下报告 paired effects。

### Weakness 3: SI-DMS 仍不能证明干预鲁棒性是内生结构属性

- **Weakness:** 所有模型在训练时都见过 `K in {0,1,2,4,8}`、15% fraction 的干预；测试外推到更大 K 是有效的 robustness-to-severity 证据，但不是“未经过干预训练也天然抵抗 reset”的证据。该任务同时触发 native reset 和下一步 recurrent propagation，也不是 pure-reset intervention。
- **Evidence basis:** Main lines 327–335；Supplement Table S3 与 Sections 3.2–3.4。
- **Reviewer deduction:** 结果支持“在同一干预课程下，SPRiF full 对更大 K 泛化更好”，但无法完全区分结构、干预训练可学习性和 recurrent dynamics 的作用。
- **Required fix:** 增加 K=0-only training 后的 intervention evaluation；再增加只改变 local reset path、但阻断或匹配 recurrent spike propagation 的配对干预，形成 `training exposure × reset path × recurrence` 的最小因子设计。

### Weakness 4: 消融与 SI-DMS 的统计证据不足

- **Weakness:** Table 2 每格只有单个数值，没有说明 ablation seed 数、标准差或统计检验；0.33–1.03 点的 lambda=0 差异可能落在运行波动内。SI-DMS 只有三种子，且未报告每个 condition 的样本数、置信区间或 model-by-stress interaction。
- **Evidence basis:** Main Tables 2–3；Supplement Sections 2.2、3.2–3.3。补充 checklist 也诚实回答没有显著性检验。
- **Reviewer deduction:** 大幅 merged drop 是可见的，但小幅 projective-reset 增益和跨模型 stress slope 尚未达到强推断标准。
- **Required fix:** 所有消融使用相同五种子；报告 mean±std/95% CI、逐种子差值；SI-DMS 明确每个 cell 的 probe 数并对 `model × K × fraction` 做配对或分层分析。

### Weakness 5: 原创性边界仍需更直接地对照最近工作

- **Weakness:** DMP-SNN、LSTM-LIF/TC-LIF 和 CLIF 都包含某种额外 memory/state/path 与 spike-generating path；当前 related work 已覆盖大部分论文，但只用简短文字强调“per-neuron”“projective reset”，没有逐结构说明 novelty delta。
- **Evidence basis:** Main Related Work lines 83–93；CLIF 已在 BibTeX 中但未在正文引用。
- **Reviewer deduction:** 原创性不至于坍塌，但“functional state decomposition”作为原则级新概念偏强。更可信的新意是特定的 `per-neuron constrained spectral slow state + reset-isolated two-dimensional fast state + learned projective reset + intervention diagnostic` 组合。
- **Required fix:** 增加一张紧凑 comparison matrix，列出 memory locus、state dimension、spectrum、spike readout、reset target、recurrence、parallelism、state/parameter cost 和 matched evidence；相应收窄 contribution 1 的原则级措辞。

### Weakness 6: 实用性与效率证据仍以参数量为主

- **Weakness:** SPRiF 每神经元存储五个状态并拥有 13 个 dynamics parameters；论文只给 trainable-parameter count 和渐近 `O(h)`/`O(h^2)`，没有 wall-clock、state-memory、MAC/add、spike rate 或能耗对比。
- **Evidence basis:** Supplement lines 423–430；Main Table 1。
- **Reviewer deduction:** “practical”可以成立为实现可行性，但不能进一步外推为神经形态效率或计算优势。
- **Required fix:** 报告每 time step 的状态内存、运算量、训练吞吐、推理延迟和平均 firing rate；若截止前无法完成，则继续避免效率型强主张。

### Weakness 7: 可复现性细节较好，但关键审计材料未全部进入本次证据链

- **Weakness:** 补充材料给出了硬件、软件、seeds、架构与优化器，但没有完整列出数据集 split cardinalities、每个 SI-DMS cell 的样本数、超参数搜索范围与选择过程；QTDB 原始数据到 `.mat` 的转换脚本明确缺失。匿名代码包被声称随投稿提交，但本次未检查其内容。
- **Evidence basis:** Supplement Sections 2.1–2.3、3.2、5；AAAI-27 明确规定“接收后再公开”不能替代投稿时提供的复现材料。[AAAI-27 Submission Instructions](https://aaai.org/conference/aaai/aaai-27/submission-instructions/)
- **Reviewer deduction:** equations 高度可复现，reported numbers 的端到端独立复现仍为中等可信度。若匿名 code/data package 实际完整上传，风险会明显下降。
- **Required fix:** 在 supplement/code README 中加入 split sizes、selection rules、evaluation sample counts、命令行、预期输出和文件 manifest；确认匿名 code/data package 与独立 checklist 均按 OpenReview 指定栏上传。

### Weakness 8: 两处 PDF 字体未嵌入

- **Weakness:** 主稿 Figure 2 的 PDF 资源含未嵌入 `ArialMT`；补充 Figure S1 含未嵌入 `Arial-BoldMT`。
- **Evidence basis:** `pdffonts` 对最终 PDF 与源 figure PDF 的检查；Poppler 渲染时也出现替代字体警告。
- **Reviewer deduction:** 这是投稿合规与跨平台显示风险，不是科学问题，但可能被自动检查拦截。
- **Required fix:** 重新导出 `si_dms_mechanism_composite.pdf` 和 `loss_landscape_smnist.pdf`，嵌入全部字体或把文字转换为可接受的矢量轮廓；重新编译后确保 `pdffonts` 的 `emb` 列全部为 `yes`。

## 7. Potentially Missing Related Work

### Work: Huang et al., “CLIF: Complementary Leaky Integrate-and-Fire Neuron for Spiking Neural Networks,” ICML 2024

- **Status:** searched
- **Why relevant:** CLIF 增加 complementary temporal path 以改善时间梯度，同时保持 binary output，属于“额外状态/路径与 spike output 分工”的近邻设计。
- **Overlap:** 多状态神经元、temporal processing、binary communication、额外路径。
- **Needed comparison:** CLIF 的 complementary path 是否接受 reset、其状态维度与梯度作用，与 SPRiF slow/fast separation 和 projective reset 的实质区别。该条目已在 BibTeX 中，但正文未引用。

### Work: Sun et al., “Algorithm–Hardware Co-Design of Neuromorphic Networks with Dual Memory Pathways,” Nature Machine Intelligence 2026

- **Status:** user-provided and searched
- **Why relevant:** 是原则层面最接近的 slow memory + fast spiking pathway 工作。
- **Overlap:** 稳定低维慢记忆、快速脉冲通路、长序列任务、参数/硬件动机。
- **Needed comparison:** 当前已加入 citation 和部分 benchmark，但还需要正式比较 layer-shared vs per-neuron、reset target、state cost、recurrence 与 matched protocol。

### Work: Zhang et al., LSTM-LIF / TC-LIF

- **Status:** user-provided and searched
- **Why relevant:** dendritic long-term memory 与 somatic discharge 的 compartmental division 与本文动机直接相邻。
- **Overlap:** slow/long memory compartment、fast/somatic discharge、长时序分类。
- **Needed comparison:** reset 是否直接作用于 memory compartment、状态更新顺序、谱参数化、梯度路径和 capacity-matched performance。

## 8. Claim-Evidence Audit

| Claim | Where stated | Evidence provided | Strength | Reviewer deduction | Required fix |
| --- | --- | --- | --- | --- | --- |
| Direct local reset 对 slow state 的同一步增量为零 | Method Eqs. 6, 12–13；Conclusion | 更新方程与 unrolled recurrence | strong | 在固定 ordinary input 的窄定义下成立；作者也披露 recurrent spike 的下一步作用 | 保持 “direct local / same-step” 限定 |
| Slow transition 谱稳定且包含实衰减与旋转模态 | Method Eqs. 7–10；Supplement Sec. 4.1 | sigmoid 参数化、eigenvalues、impulse response | strong | 数学结论清楚且可验证 | 可补充边界参数接近 1 时的数值稳定性 |
| 在列出方法中，三任务 mean accuracy 最高 | Abstract、Table 1、Conclusion | 五种子 SPRiF 均值；跨论文 baselines | adequate as descriptive / weak as superiority | 字面上成立且已正确限定为 listed/descriptive；不能解释为 matched SOTA | 增加同协议重训基线，继续保留限定语 |
| 参数量低于最高准确率 listed baseline 的四个任务 | Introduction、Main Results | Table 1 参数数目 | weak | counting convention 与跨论文 architecture 不统一，不能形成强效率结论 | 给出脚本化 counting rule 与 matched parameter-budget comparison |
| Functional state decomposition 保护 temporal context | Abstract、SI-DMS、Conclusion | merged ablation、SI-DMS、结构恒等式 | weak-to-adequate | 结构隔离成立，功能收益受容量、干预训练和 recurrence 混淆 | capacity-matched control + K=0-only training + factorial intervention |
| Learned projective reset 有 task-specific benefit | lambda=0 ablation、SI-DMS、lambda distributions | 三任务小幅 accuracy drop；SI-DMS K=40 差 4.0 点；lambda 多样性 | adequate but statistically incomplete | SI-DMS 支持较强，普通基准的小差异缺乏多种子不确定性 | multi-seed paired ablation 与 CI |
| Learned temporal kernels heterogeneous and long-lived | Figure 4、Dynamical Analysis | alpha-quantile trajectories、spectra、两任务 ASRNN diagnostic | adequate | 描述性结论得到支持；作者已避免泛化为普遍 cross-model advantage | 报告 checkpoint/seed sensitivity 或保持当前有限范围 |
| SPRiF 是 practical neuron-level design | Abstract、Conclusion | 五任务、固定维度 `O(h)` local update、SI-DMS | mixed | 可实现性与任务覆盖支持“practical”；缺少 runtime/state-memory/hardware 证据 | 添加成本测量或进一步收窄实践性措辞 |

## 9. Experiment / Benchmark / Reproducibility Audit

- **Baselines:** 文献覆盖较广，DMP-SNN 已补入，但主基准仍为 protocol-mismatched cross-paper comparison；缺少统一训练下的强基线。
- **Ablations:** omega=0、merged、lambda=0 与三项设计选择对应良好；merged 改变容量，且所有变体缺少清晰的 seed/variation 报告。
- **Datasets/benchmarks:** 五个任务具有模态多样性，但 S-/PS-MNIST 已较饱和；更具现实长度、噪声和规模的任务会增强意义。SI-DMS 是有价值但合成的机制任务。
- **Metrics:** accuracy 合理；SI-DMS 同时报 clean accuracy 与 paired loss 是正确做法。缺少 calibration、energy/latency 和统计 interaction。
- **Statistical rigor:** 主 SPRiF 结果五种子并报告 std；SI-DMS 三种子；消融无不确定性；没有显著性检验或置信区间。
- **Robustness/failure cases:** SI-DMS 有 K 和 fraction sweep；feature perturbation 与 loss landscape 被正确限定为 fixed-checkpoint/single-slice diagnostics。仍缺少失败案例和无干预训练的 stress test。
- **Implementation details:** equations、任务架构、optimizer、learning rates、thresholds、initialization ranges、surrogate gradient、compute environment 和 seeds 较完整。
- **Artifacts and reproducibility:** 独立 checklist 已填写；supplement 声称匿名 code archive 随投稿提供。代码包、原始逐种子表、数据预处理完整性未在本次审查中验证。
- **Limitations:** 作者已披露多个关键边界，表现良好；仍应显式加入 unmatched baselines、synthetic SI-DMS、intervention curriculum、无硬件/能耗验证和统计功效限制。

## 10. Multi-Reviewer Panel

### Reviewer: Best-Justified Reviewer

- **Expertise:** temporal SNN neuron design
- **Likely score:** 7/10
- **Confidence:** 4/5
- **Main positive signal:** 具体构造新颖、方程干净、稳定且解释性强；SI-DMS 和 scope qualifiers 显示出良好研究判断。
- **Main negative signal:** matched empirical evidence 仍不足。
- **Score-change condition:** capacity-matched separation control 和三任务 matched baselines 可把 accept case 变得稳定。

### Reviewer: Critical Reviewer

- **Expertise:** empirical machine learning methodology
- **Likely score:** 4/10
- **Confidence:** 4/5
- **Main positive signal:** 模型和实验均可理解，作者没有掩盖明显限制。
- **Main negative signal:** headline benchmark 不能归因于 neuron，核心 merged control 又被容量混淆。
- **Fatal concern if any:** 无单一已证实的 fatal flaw；若匿名代码包缺失或字体检查失败，可能转化为合规性问题。
- **Score-change condition:** 同协议基线、多种子消融、容量匹配 control。

### Reviewer: Method / Soundness Reviewer

- **Expertise:** dynamical systems and spiking neurons
- **Likely score:** 6/10
- **Confidence:** 4/5
- **Main positive signal:** 谱稳定性、update order 和 reset identity 都是清楚且正确的。
- **Main negative signal:** 结构恒等式被外推为功能性记忆保护，而因果实验尚未完全隔离该机制。
- **Score-change condition:** 把主张严格限定为 structural insulation，或提供只改变 reset target 的 capacity-matched control。

### Reviewer: Evidence / Experiment Reviewer

- **Expertise:** benchmark design and statistical evaluation
- **Likely score:** 4/10
- **Confidence:** 5/5
- **Main positive signal:** 五任务、机制变体、干预任务、完整 fraction grid 的证据包较丰富。
- **Main negative signal:** unmatched Table 1、无误差条的 Table 2、三种子 SI-DMS 和 intervention-trained setting。
- **Score-change condition:** matched baselines、五种子 ablations、明确 condition sample size 与 interaction analysis。

### Reviewer: Novelty / Positioning Reviewer

- **Expertise:** SNNs and state-space sequence models
- **Likely score:** 5/10
- **Confidence:** 4/5
- **Main positive signal:** per-neuron spectral slow state、two-dimensional fast discharge 与 learned reset direction 的组合有辨识度。
- **Main negative signal:** functional decomposition 原则与 DMP-SNN、LSTM-LIF/TC-LIF、CLIF 存在明显先行脉络。
- **Score-change condition:** comparison matrix 与收窄后的 novelty claim。

### Reviewer: Writing / Clarity Reviewer

- **Expertise:** scientific exposition
- **Likely score:** 7/10
- **Confidence:** 4/5
- **Main positive signal:** 贡献、方程、图表和限制在一次仔细阅读后可以恢复，术语基本稳定。
- **Main negative signal:** Related Work 对 closest-work delta 过于压缩，部分图中文字偏小。
- **Score-change condition:** 增加结构对照表并进一步提升 Figure 2/3 的最小字号。

### Reviewer: Ethics / Reproducibility Reviewer

- **Expertise:** reproducible ML and submission compliance
- **Likely score:** 5/10
- **Confidence:** 4/5
- **Main positive signal:** 公共数据、匿名材料、完整 compute environment、明确 seeds、独立 checklist 和声称随投的匿名代码包。
- **Main negative signal:** 代码包未核验、QTDB converter 缺失、split/sample-count/search protocol 不完整、两处字体未嵌入。
- **Score-change condition:** 修复字体并提供可独立运行的 code/data manifest。

### Reviewer: AC / Meta-Reviewer

- **Expertise:** broad AAAI machine learning
- **Likely score:** 5/10
- **Confidence:** 4/5
- **Main positive signal:** 是一篇可理解、可能有用、机制动机明确的 neuron-design paper，不是单纯堆 benchmark。
- **Main negative signal:** 两位最关键 reviewer——evidence 与 novelty——都可能停在 negative side，且其 concern 不能靠 rebuttal 文字化解。
- **Score-change condition:** 投稿前补齐 matched evidence 和 capacity-matched causal control；仅润色不会改变决定轴。

## 11. Concerns Table

| ID | Severity | Concern | Evidence basis | Affected criterion | Fix class | Required action | Owner skill | Score-change condition |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C1 | major | 主基准无同协议重训基线 | Table 1 明确为 cross-paper descriptive comparison | evidence, significance | experiment | 三个代表任务上重训 matched baselines | `ccf-experiment-designer` | Evidence 可从 2 升至 3–4 |
| C2 | major | merged control 改变总状态维度与 clean capability | Main Table 2；Supplement Table S2；SI-DMS clean 88 vs 100 | soundness, evidence | experiment | five-state capacity-matched merged control | `ccf-experiment-designer` | 核心 separation claim 才可被因果支持 |
| C3 | major | SI-DMS 训练见过干预且同时测 reset+recurrence | Main Sec. Controlled Spike Intervention；Supplement Table S3 | soundness, evidence | experiment | K=0-only training 与 factorial intervention | `ccf-experiment-designer` | 内生 robustness 结论才可成立 |
| C4 | major | ablation 无 seed/variance；SI-DMS 统计功效有限 | Main Tables 2–3；checklist 无统计检验 | evidence | experiment | 五种子、CI、paired effects、interaction | `ccf-experiment-designer` | lambda 与小幅增益可被可信解释 |
| C5 | moderate | 原则级 novelty 与最近工作重叠 | DMP-SNN、LSTM-LIF/TC-LIF、CLIF、SpikingSSMs | originality, positioning | related-work | closest-work comparison matrix，收窄 claim | `ccf-literature-search` | Originality 可稳定在 3–4 |
| C6 | major | 最终 PDF 含未嵌入字体 | Figure 2 `ArialMT`；Figure S1 `Arial-BoldMT` | compliance | reproducibility | 重导 figure 并确保所有字体 embedded | `ccf-submission-checker` | 消除 desk/upload 风险 |
| C7 | moderate | 缺少 runtime/state-memory/energy 对比 | 每神经元 5 states、13 dynamics params；仅给 parameter count | significance | experiment | 报告 memory、ops、throughput、latency、spike rate | `ccf-experiment-designer` | 强化 practical/efficiency 价值 |
| C8 | moderate | 端到端复现协议仍有空白 | split sizes、SI-DMS cell size、hyperparameter search、QTDB converter | reproducibility | reproducibility | 完整 code/data manifest 与 commands | `ccf-integrity-auditor` | Reproducibility 可从 3 升至 4 |
| C9 | moderate | CLIF 未在正文定位 | BibTeX 有 `clif`，Related Work 未引用 | originality, clarity | related-work | 补一到两句机制差异 | `ccf-writing-skills` | 降低 omission 风险 |
| C10 | minor | Supplement pages 3/9 留白较大，部分 figure labels 偏小 | PDF visual review | clarity | writing | 重新安排 float/page breaks；放大最小字号 | `ccf-conference-writing-reviewer` | 单独不改变总分 |

## 12. AC / Meta-Review

- **Reviewer consensus:** 数学构造清楚，direct local reset insulation 的窄结论成立；论文有实际研究价值，图表和限制性措辞优于普通 neuron-paper。现有证据还不能把主基准优势和核心 state-separation 收益稳健归因于 SPRiF 机制。
- **Reviewer disagreement:** method/writing reviewer 可能给 weak accept，认为具体组合足够新且证据包丰富；experiment/novelty reviewer 会因 unmatched baselines、capacity confound 和 intervention curriculum 给 weak reject。
- **Decisive acceptance axis:** 是否存在容量匹配、只改变 reset-to-memory path 的强 control，以及在统一协议下 SPRiF 是否稳定优于强基线。
- **Decisive rejection axis:** 若 Table 1 仍是唯一主性能证据、Table 2 仍无 uncertainty、merged 仍不匹配容量，核心贡献会被评价为“合理但未被决定性验证”。
- **AC stance:** borderline negative / weak reject，Overall 5/10。
- **Discussion risks:** AAAI 两阶段评审下，evidence reviewer 的 major concerns 足以造成 Phase-1 淘汰；即使进入讨论，novelty reviewer 也会要求与 DMP-SNN/TC-LIF/CLIF 更明确的结构差异。字体问题应在投稿前解决，避免科学讨论之外的合规损失。

## 13. Scores

- **Quality:** 3/5
- **Clarity:** 4/5
- **Significance:** 3/5
- **Originality:** 3/5
- **Soundness:** 3/5
- **Evidence:** 2/5
- **Reproducibility:** 3/5
- **Ethics / Limitations:** 4/5
- **Overall:** 5/10 — borderline negative / weak reject
- **Confidence:** 4/5

该评分是基于当前材料的 review-risk diagnosis，不是接收概率。最大扣分来自证据设计，而不是认为 SPRiF 方程错误。

## 14. Questions For Authors

1. Table 2 的 omega=0、merged、lambda=0 各使用多少独立种子？逐种子结果和标准差是多少？
2. 能否构造一个保持五个状态、相近参数量和相近 clean accuracy 的 merged/non-separated control，只让 reset 进入 memory-carrying coordinates？
3. 在完全不含 spike intervention 的 K=0-only training 后，SPRiF、ASRNN、BRF 和 capacity-matched merged model 的 SI-DMS 曲线如何？
4. 在 PS-MNIST、GSC、SHD 上，使用完全相同 architecture、optimizer、epochs、seeds 和 readout 重训强基线后，SPRiF 的逐种子 paired difference 是否仍为正？
5. SI-DMS 每个 seed/delay/K/fraction condition 包含多少 evaluation trials？是否可以报告置信区间及 `model × stress` interaction？
6. 与 DMP-SNN、TC-LIF/LSTM-LIF 和 CLIF 相比，哪些状态直接接受 reset，哪些状态承载 memory，以及各自每 neuron/layer 的 state cost 是多少？
7. 最终匿名 code/data package 是否包含所有五任务的 preprocessing、split manifest、evaluation commands 和 expected hashes/results；QTDB 转换缺口如何处理？
8. 能否在提交前重新导出 Figure 2 和 Figure S1，使最终两个 PDF 的所有字体均嵌入？

## 15. Score Revision Criteria

**Raising the score would require:**

- three-task matched-baseline evaluation with multi-seed paired reporting；
- five-state capacity-matched merged control；
- multi-seed ablations with uncertainty；
- K=0-only SI-DMS training control and a design separating local reset from recurrent propagation；
- direct closest-work comparison matrix and narrower principle-level novelty wording；
- embedded-font-clean PDFs and complete anonymous code/data manifest。

若前四项给出一致的正结果，Overall 6–7/10 的 borderline positive / weak accept 立场将有证据基础；这不是对分数变化的保证。

**Lowering the score would be triggered by:**

- matched baselines 消除当前性能差异；
- capacity-matched merged control 与 SPRiF 表现相当；
- K=0-only training 后 SPRiF 的 intervention robustness 消失；
- code/data package 无法复现关键表格，或提交 PDF 仍有字体/匿名性问题；
- 更接近的同行评审工作使 spectral/projective-reset 组合也不再具有清晰 novelty delta。

**Concerns unlikely to change before submission:**

- SI-DMS 仍是合成机制任务；
- 完整神经形态硬件与能耗评估可能无法在截稿前完成；
- functional decomposition 不能再作为完全无先例的原则，只能围绕 SPRiF 的具体实现建立原创性。

## 16. Action Plan And CCFA Handoffs

### Action 1

- **Priority:** P0
- **Action:** 修复 Figure 2/Figure S1 字体嵌入并重做最终 PDF 合规检查。
- **Owner skill:** `ccf-submission-checker`
- **Input needed:** 两个 figure 源文件、最终主稿与补充 PDF
- **Expected output:** 全字体 embedded、页数/匿名性/编译均通过的提交包
- **Handoff required:** yes

### Action 2

- **Priority:** P0
- **Action:** 设计并运行 capacity-matched merged control 与五种子机制消融。
- **Owner skill:** `ccf-experiment-designer`
- **Input needed:** SPRiF/merged 实现、现有训练配置、可用算力
- **Expected output:** capacity-controlled Table 2、逐种子结果与 CI
- **Handoff required:** yes

### Action 3

- **Priority:** P0
- **Action:** 在代表性任务上补齐 matched baselines。
- **Owner skill:** `ccf-experiment-designer`
- **Input needed:** LIF/ASRNN/BRF/TC-LIF 实现、统一训练协议
- **Expected output:** 可归因的 matched benchmark table
- **Handoff required:** yes

### Action 4

- **Priority:** P1
- **Action:** 增加 K=0-only SI-DMS 和 reset/recurrence 因子控制。
- **Owner skill:** `ccf-experiment-designer`
- **Input needed:** SI-DMS 代码、intervention hooks、训练预算
- **Expected output:** 结构内生 robustness 与训练适应效应的分离结果
- **Handoff required:** yes

### Action 5

- **Priority:** P1
- **Action:** 完成 closest-work comparison matrix，并收窄 contribution/abstract/conclusion 的原则级新颖性措辞。
- **Owner skill:** `ccf-literature-search` followed by `ccf-writing-skills`
- **Input needed:** 当前 Related Work、DMP-SNN/TC-LIF/LSTM-LIF/CLIF/SpikingSSMs
- **Expected output:** 结构对比表和 claim-evidence-aligned 文本
- **Handoff required:** yes

### Action 6

- **Priority:** P1
- **Action:** 补全匿名 code/data manifest、split/sample-count、运行命令和统计报告。
- **Owner skill:** `ccf-integrity-auditor`
- **Input needed:** 提交代码包、原始 logs/results、数据准备说明
- **Expected output:** 可审计复现包和数字一致性 ledger
- **Handoff required:** yes

**Checks run:** AAAI-27 venue/track/rules；主稿与补充材料四遍阅读；贡献图谱；desk checks；页数与 US-Letter 检查；匿名性扫描；prompt-injection/hidden-manipulation 扫描；LaTeX clean build；17 页 PDF 视觉检查；字体嵌入检查；claim-evidence audit；baseline/ablation/SI-DMS/statistics/reproducibility audit；公开安全 closest-work 搜索；多审稿人模拟；AC synthesis；score consistency check；独立 reproducibility checklist 完成状态核验。

**Checks skipped:** 全量源代码审计；重新训练或独立复现实验；从 raw per-seed outputs 重算所有均值/标准差；匿名 code/data archive 的上传内容与可运行性验证；穷尽全部 2026 非同行评审预印本。

**Unresolved risks:** Table 2 的真实 seed 数与方差；matched-protocol 下的相对性能；capacity-matched separation effect；K=0-only training 下的 SI-DMS robustness；匿名代码包能否端到端复现；修复字体后的最终 PDF 状态。
