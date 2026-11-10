# TNNLS 官方 LaTeX 模板

下载日期：2026-10-03。

模板来自 IEEE 官方 Template Selector，选择路径为：

**Transactions, Journals and Letters → IEEE Transactions on Neural Networks and Learning Systems → Original research and Brief → LaTeX。**

选择器明确将这个模板包用于 TNNLS 的原创研究论文和 Brief。它使用 IEEE Transactions 通用的 IEEEtran 类，并不是另一个名为 TNNLS 的专用文档类。

## 文件

| 文件 | 用途 |
|---|---|
| `IEEE-Transactions-LaTeX2e-templates-and-instructions.zip` | 官方原始压缩包，保留未修改版本 |
| `official/bare_jrnl_new_sample4.tex` | 期刊论文示例源文件，可复制后作为新稿起点 |
| `official/bare_jrnl_new_sample4.pdf` | 官方提供的示例 PDF |
| `official/IEEEtran.cls` | 文档类；包内标注版本 V1.8b |
| `official/New_IEEEtran_how-to.tex`、`.pdf` | LaTeX 使用说明 |
| `official/fig1.png` | 示例图片 |
| `source.json` | 官方选择结果、下载链接及下载日期 |

示例的文档类声明是：

```latex
\documentclass[lettersize,journal]{IEEEtran}
```

## 使用

在 Overleaf 中可以选择 New Project → Upload Project，直接上传原始 ZIP，并将 `bare_jrnl_new_sample4.tex` 设为主文件。后续另存或复制示例再写 SPRiF 稿件，保留本目录的官方原件。

使用本地 LaTeX 时，将主文件、`IEEEtran.cls`、图片和参考文献文件放在可正确引用的位置。包内示例使用手工参考文献；若改为现有 `.bib` 工作流，需要相应的 IEEEtran BibTeX 样式及正常引用编译流程。

本次已确认下载返回 ZIP，并检查和解压全部六个官方文件。示例 PDF 为下载包自带文件，本次没有重新编译模板，也没有迁移或修改 `AuthorKit27` 中的论文。

模板负责排版，TNNLS 的页数、费用及提交材料要求仍需按投稿时的作者指南核实。

## 官方来源

- [IEEE Author Center 模板说明](https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/authoring-tools-and-templates/tools-for-ieee-authors/ieee-article-templates/)。
- [IEEE Template Selector](https://template-selector.ieee.org/)。
- [TNNLS 模板选择结果](https://template-selector.ieee.org/api/ieee-template-selector/template/publication-type/1/publication-title/84/article/3)。
- [所选 LaTeX 包的官方直接下载](https://template-selector.ieee.org/api/ieee-template-selector/template/262/download)。
