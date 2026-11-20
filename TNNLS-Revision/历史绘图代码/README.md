# TNNLS 配图历史代码

整理日期：2026-11-01；补充核对：2026-11-05。

此目录集中保存与 TNNLS 配图相关的历史绘图、数据导出和排版脚本，供后续适配新模型使用。Python 脚本和依赖清单均保留来源文件内容；当前模型示意图另以 TikZ 片段保存。这里尚未进行新版模型接口、五种子统计或路径适配。

当前 TNNLS 正文实际引用一张 SPRiF 结构图。时间核、发放前后轨迹和 SI-DMS 曲线是可根据论文论证及可用数据选择的后续配图，以下编号用于查找代码，不代表论文已有四张图。

## 按图查找

| 图的用途 | 先查看的代码 | 其他相关代码 |
|---|---|---|
| 结构示意图 | [current_tnnls_diagram.tikz](01-structure/current_tnnls_diagram.tikz) | 当前主稿 SPRiF 状态和单步更新示意图 |
| 学习到的时间核及读出通路 | [impulse_analyze.py](02-temporal-kernels/impulse_analyze.py) | [plot_combined_figures.py](05-shared-composites/plot_combined_figures.py) 中的 `build_temporal_figure` |
| 发放前后的状态与复位响应 | [trajectory_analyze.py](03-reset-mechanism/trajectory_analyze.py) | [plot_combined_figures.py](05-shared-composites/plot_combined_figures.py) 中的 `build_mechanism_figure`，以及精修版 [redraw_figure2.py](05-shared-composites/redraw_figure2.py) |
| SI-DMS 干预曲线和脉冲栅格 | [plot_si_dms_composite.py](04-si-dms/plot_si_dms_composite.py) | [plot_results.py](04-si-dms/plot_results.py)、[plot_raster.py](04-si-dms/plot_raster.py)、[aggregate.py](04-si-dms/aggregate.py) |

## 文件与来源

以下八个 Python 文件和一份依赖清单复制自工作区，保存了来源文件内容。

| 本目录文件 | 仓库内来源路径 |
|---|---|
| `02-temporal-kernels/impulse_analyze.py` | `代码/experiments/impulse_analysis/impulse_analyze.py` |
| `03-reset-mechanism/trajectory_analyze.py` | `代码/experiments/trajectory_analysis/trajectory_analyze.py` |
| `03-reset-mechanism/reset_analyze.py` | `代码/experiments/reset_analysis/reset_analyze.py` |
| `04-si-dms/plot_si_dms_composite.py` | `experiment-design-20260606/results/figures/si_dms/plot_si_dms_composite.py` |
| `04-si-dms/test_plot_si_dms_composite.py` | `experiment-design-20260606/results/figures/si_dms/test_plot_si_dms_composite.py` |
| `04-si-dms/plot_results.py` | `代码/experiments/SI-DMS/plot_results.py` |
| `04-si-dms/plot_raster.py` | `代码/experiments/SI-DMS/plot_raster.py` |
| `04-si-dms/aggregate.py` | `代码/experiments/SI-DMS/aggregate.py` |
| `04-si-dms/requirements.txt` | `代码/experiments/SI-DMS/requirements.txt` |

`05-shared-composites/` 中的五个 Python 文件从 Git 提交 `5448f15ce2aa3d2e9a7db5fc8bef2418e8b91040` 恢复，来源目录为 `代码/experiments/combined_analysis/`。

- `plot_combined_figures.py` 生成原 `mechanism_composite` 和 `temporal_kernels_composite`，并导出 PDF、SVG、PNG。
- `redraw_figure2.py` 生成机制综合图的后续精修版 `mechanism_composite_v5`。
- `test_plot_combined_figures.py` 保存历史排版和源数据检查。
- `check_figure_contrast.py` 检查历史配色对比度。
- `verify_figure_quality.py` 检查历史时间核图的字体、分辨率和灰度表现。

`01-structure/current_tnnls_diagram.tikz` 从整理时的 [TNNLS 主稿](../../TNNLS-Manuscript/SPRiF_TNNLS.tex) 原样提取，内容范围为 `tikzpicture` 环境。2026-11-05 已核对其与当前主稿的绘图部分一致。尚未找到 [AAAI 原架构图 PNG](../../AuthorKit27/figures/lif_sprif_overview.png) 的独立生成脚本。TikZ 片段需要宿主文档加载 `tikz`、`arrows.meta`、`positioning`、`fit`，并定义 `\R` 为 `\mathbb{R}`。

## 改成新版图片前需要处理的地方

1. **路径。** 副本保留了原路径逻辑。部分代码用 `__file__` 的父目录计算仓库根目录，另有脚本硬编码旧机器路径；搬到本目录后不能直接按原逻辑运行。先改成显式输入、输出参数。
2. **输入数据。** 时间核与机制综合图读取 `trajectory_data.npz`、`trajectory_data_phi1.npz`、`raw_impulse_responses.npz`、`asrnn_comparison_data.npz` 和 `lambda_stats.csv` 等历史文件。当前目录只整理代码，后续需从新版 checkpoint 导出与新面板对应的数据。
3. **训练触发。** `impulse_analyze.py`、`trajectory_analyze.py` 和 `reset_analyze.py` 中包含找不到 checkpoint 时启动训练的分支。适配为绘图工具时，先改为要求显式传入现有 checkpoint，缺失就报错。
4. **新版状态。** 轨迹提取需按 `x`、`h`、标量 `u` 及复位前后时点适配。膜候选读取 `h_(t-1)`，新图需要保留这一延迟。
5. **SI-DMS 统计。** 原综合脚本读取旧模型组和三个种子，并依赖旧实验目录。适配时应使用 SPRiF、MEMORY_RESET5、ASRNN、LIF、BRF、五种子及 Clean / Intervention 两种训练条件。绘制 K 或 delay 曲线需对应条件的评估结果，不能从表3补造。历史 `aggregate.py` 和 `plot_raster.py` 还依赖原 [SI-DMS 模块](../../代码/experiments/SI-DMS/sidms)及其配置，复制到本目录后不能视为独立可运行程序。
6. **旧 λ 面板。** `reset_analyze.py` 仅供追溯旧机制图的数据来源。新模型没有可学习 λ，最终图片去掉 λ 分布、复位方向扇形和相关性面板。
7. **历史测试。** 两份测试文件保留了旧的模型组、种子数、状态尺寸和路径假设，需随新图重写。`redraw_figure2.py` 中一个旧字符串的反斜杠在较新 Python 中会产生 `SyntaxWarning`，适配时再修正。

## 已完成的整理校验

- 13 个 Python 文件及依赖清单均已与工作区来源或指定 Git 对象逐字节核对。
- 13 个 Python 文件均通过语法解析。
- TikZ 片段与当前主稿中的模型图代码一致。
- 本次未导入或运行上述脚本，因此没有启动训练、执行回放、生成新实验数据或重绘图片。
