# SPRiF TNNLS 正文工作稿

**更新：2026-10-05。** 本目录现为采用 IEEEtran 期刊版式的单一正文。模型定义、必要推导、实验协议、结果表和局限性均在正文中；最终论文提交使用主文稿 PDF。AuthorKit27 原稿保持原状。

| 文件 | 用途 |
|---|---|
| [SPRiF_TNNLS.tex](/C:/Users/VECTOR/Desktop/SPRiF-Neuron/TNNLS-Manuscript/SPRiF_TNNLS.tex) | 当前主文稿；参考文献和模型示意图均写在此源文件内 |
| [SPRiF_TNNLS.pdf](/C:/Users/VECTOR/Desktop/SPRiF-Neuron/TNNLS-Manuscript/SPRiF_TNNLS.pdf) | 当前主文稿编译结果，共 10 页 |
| [sprif2027.bib](/C:/Users/VECTOR/Desktop/SPRiF-Neuron/TNNLS-Manuscript/sprif2027.bib) | 作者维护的文献库；新增四篇 TNNLS 文献，正文内嵌参考文献需同步维护 |
| [IEEEtran.cls](/C:/Users/VECTOR/Desktop/SPRiF-Neuron/TNNLS-Manuscript/IEEEtran.cls) | 已下载官方模板中的期刊文档类，用于编译 |
| [正文重构与核对记录](/C:/Users/VECTOR/Desktop/SPRiF-Neuron/TNNLS-Revision/10-正文重构与核对记录-20261005.md) | 章节调整、文献对应关系、占位符及检查结果 |
| [步骤 01 修改记录](/C:/Users/VECTOR/Desktop/SPRiF-Neuron/TNNLS-Revision/08-步骤01修改记录.md) | 此前模型修改的历史记录 |

`SPRiF_TNNLS_supp.tex` 是先前工作备份。必要内容已迁入正文，当前主文稿不引用、不加载它，提交论文不需要该文件。重构前的主文源文件保存在 `build/SPRiF_TNNLS-before-restructure.tex`。

## 章节顺序

1. Introduction
2. Related Work
3. SPRiF Neuron: Model and Analysis
4. Experimental Setup
5. Results and Discussion
6. Conclusion

## 新模型

\[
\begin{aligned}
\mathbf{x}_t&=A\mathbf{x}_{t-1}+BI_t,\\
h_t&=\eta_1h_{t-1}+\bar{\mathbf g}_1^\top\mathbf{x}_t,\\
\widetilde u_t^0&=\eta_0u_{t-1}^0+h_{t-1}+\mathbf g_0^\top\mathbf{x}_t,\\
z_t&=H(\widetilde u_t^0-\theta),\\
u_t^0&=\widetilde u_t^0-z_t\theta.
\end{aligned}
\]

三维 x、标量 h、标量 u⁰，共 **5 状态**。α、ρ、ω、η₀、η₁ 和 bar_G 的六个条目，共 **11 个动力学学习参数**。θ 固定，突触、偏置和输出参数另计。膜候选使用 **h_(t−1)**。

| 统一术语 | 对应符号/含义 | 原稿表述的处理 |
|---|---|---|
| spectral state | 三维 x，一实模态和二维旋转块 | 保留三维谱参数化 |
| auxiliary filter | 标量 h，不受直接复位 | 取代原第二快坐标的解释 |
| membrane coordinate | 标量 u⁰，标量减法复位 | 不再称两维快状态都承担复位 |
| memory and filter pathway | 四维 (x,h) | 不强制其全部时间常数比膜更慢 |
| readout matrix | bar_G=[g₀ᵀ; bar_g₁ᵀ] | 吸收 κ，六个条目直接学习 |

λ 从当前主文稿中删除。κ 仅在正文 III-C 的旧模型转换推导中出现，不是新模型学习参数。历史 Full（13 参数）和 scalar-reset（12 参数）指标在 V-A 明确说明归属；当前 11 参数模型的主结果、控制实验、SI-DMS、初始化敏感性及成本结果均为 TBD。

## 后续边界

本次完成正文结构、模型与分析、协议说明及结果讨论框架。未修改实现、未启动训练。实际 bar_G 初始化、数据划分、读出细节和部分运行设置仍待冻结；原稿给出的训练配置被标为候选起点。旧经验图片没有移入新模型结果，正文模型示意图已按新定义重绘。补齐新实验和协议字段后，才能形成可正式提交的实证稿件。

## 编译

在本目录执行：

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build SPRiF_TNNLS.tex
Copy-Item -LiteralPath 'build/SPRiF_TNNLS.pdf' -Destination 'SPRiF_TNNLS.pdf' -Force
```

本次使用现有本地 TeX Live 编译成功。正文有 32 个编号公式、10 个表、1 幅图和 17 条参考文献；无未解析引用、重复标签或溢出版面。内置 LaTeX 编译器报运行环境错误，具体信息见核对记录。旧 AAAI 样式文件留作历史资源，当前正文不加载它们。
