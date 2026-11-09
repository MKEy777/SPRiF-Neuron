# 参考文献恢复与表 I 调整（2026-11-06）

正文引用从 23 篇增至 35 篇，保留原有全部引用，恢复 12 篇。修订集中在 Introduction 与 Related Work；没有改变模型公式、实验准确率或原始图片。

## 恢复的讨论与核对来源

| 方向 | 引用键 | 本文采用的论点 | 核对来源 |
|---|---|---|---|
| 时间尺度 | dalif | 独立可学习衰减用于时空调节 | https://arxiv.org/abs/2502.10422 |
| 时间尺度 | bimodal | 短、长时间尺度神经元群体 | https://www.sciencedirect.com/science/article/pii/S0893608026002923 |
| 时间尺度 | hetsyn | 突触特异时间常数与突触电流积分 | https://openreview.net/pdf/0ab977c647bf52844f3c46304c13a0a958674a61.pdf |
| 时间尺度 | mtc | 快、慢、超慢电导组织动态 | https://arxiv.org/abs/2605.11835 |
| 共振 | prf | 共振膜动态与解耦复位的并行计算 | https://arxiv.org/abs/2410.03530 |
| 树突 | dendrify | 简化分室模型中的树突积分和局部脉冲 | https://www.nature.com/articles/s41467-022-35747-8 |
| 状态空间 | silif | SSM 启发的离散化、参数化与振荡动态 | https://arxiv.org/abs/2506.06374 |
| 复位 | inflor | 软复位与膜电位校正减少信息损失 | https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/88_ECCV_2022_paper.php |
| 复位 | arlif | 通过输入、输出关系自适应复位 | https://arxiv.org/abs/2507.20746 |
| 不应期 | rplif | 脉冲触发的阈值动态 | https://arxiv.org/abs/2509.17769 |
| 复位分析 | zhang2025reset | 二值激活 RNN 视角下的复位和不应期 | https://arxiv.org/abs/2504.17751 |
| 无复位计算 | psn | 移除复位得到可并行的时间计算 | https://proceedings.neurips.cc/paper_files/paper/2023/hash/a834ac3dfdb90da54292c2c932c997cc-Abstract-Conference.html |

采用原文摘要、官方论文页或论文文本核对上述方法概述，不新增这些工作的性能数字。原有 23 篇引用的全部元数据未作独立重审。

旧稿其余四个引用键未机械补回：lstmlif 所处的两分室方向已有正式发表的 TC-LIF 讨论；ltgate 侧重持续学习；stochastic_gating 与 neuromorphic_robustness 侧重网络鲁棒性，当前段落围绕神经元状态与复位。其文献库记录仍保留。

## 文献管理

- 正文移除手工内嵌 bibliography，改用 sprif2027.bib 和 IEEEtran.bst。
- IEEEtran.bst 来自 https://mirrors.ctan.org/macros/latex/contrib/IEEEtran/bibtex/IEEEtran.bst 。
- VS Code 默认配方和 build-manuscript.ps1 均执行 pdfLaTeX → BibTeX → pdfLaTeX → pdfLaTeX。
- 最后一次编译：35 条引用全部解析；无未定义引用、重复标签或横向溢出。

## 表 I

- 跨双栏，位于第 6 页。
- Source 改为 Network，显示 Rec 或 DMP。
- Method 后附作者名和数字引用。
- 参数列由 M 改为 K，逐行乘以 1000；38 行参数总量、准确率与标准差核对一致。
- R 表示复现，N 表示 DMP-SNN 原架构运行，未标记的基线为原文报告结果。
- 最新主稿共 10 页；参考文献末页已分配为双栏。
