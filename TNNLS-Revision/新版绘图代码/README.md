# 新版 TNNLS 组合图

本目录的三个子目录分别对应机制图、SI-DMS 和时间核图；脚本默认从各自的 `原始数据/` 读取，输出到 `输出图片/`。缺少真实数据时会报错，不会自动训练或生成替代结果。

```text
01-机制分析/   export_trajectories.py、export_benchmark_trajectories.py、plot_mechanism.py、原始数据/、输出图片/
02-SI-DMS/     plot_sidms.py、legacy_layout.py、原始数据/、输出图片/
03-时间核/     export_asrnn_parameters.py、plot_kernels.py、原始数据/、输出图片/
export_parameters.py     导出新版 SPRiF 参数
checkpoint_io.py         检查点读取和来源记录
publication.py、_vendor/ 共用出版图设置和面板对齐检查
04-训练代码/            两张图所需模型的训练入口
```

两张图所需模型的最少训练与直接绘图流程见 [训练说明](04-训练代码/README.md)。机制图与时间核图的数据格式和手动导出说明分别见两者 `原始数据/README.md`。机制图保持旧版上下两行、左/中/右三列的组合布局；原版依赖 λ 的面板换为新版状态复位量。时间核图保留旧版上方三幅时域、下方三幅频域及右侧 ASRNN 对比的布局。

```bash
python "01-机制分析/plot_mechanism.py"
python "02-SI-DMS/plot_sidms.py"
python "03-时间核/plot_kernels.py"
```

默认直接读取 `04-训练代码/checkpoints/` 下对应模型的正式 checkpoint，自动生成绘图所需输入；也支持 `--input` 或 `--manifest` 读取事先导出的数据。绘图依赖 numpy、matplotlib；从模型 checkpoint 导出数据还需训练环境已有的 torch 及其项目依赖。绘图不会自动训练模型。

参数导出支持原始 state_dict 和 `model`、`state_dict`、`model_state_dict` 包装。可自动识别唯一的层前缀；当存在多个候选层时使用 `--prefix`。旧版含 λ/κ 的 checkpoint 不能直接作为新版参数。`REAL_MODES3` 的参数槽含义不同，不能作为 SPRiF 旋转模态导出。

**证据范围：**机制脚本支持当前 SI-DMS 两个独立训练网络在同一试次输入和干预下的状态回放，也提供 PS-MNIST / SHD 当前层实现的重建回放。后者需要真实预处理后的测试样本，以及两个独立训练 checkpoint；脚本按照论文定义显式施加 MEMORY_RESET5 规则，因此必须核对 checkpoint 的实际训练来源和控制规则。轨迹包含各自网络的循环反馈，同一编号的神经元不代表两模型中功能相同。一条样本的轨迹不能替代表 2 的性能统计。时间核采用零初始状态、无发放和无复位的局部单位电流响应；右侧比较两模型的无发放膜电位响应，并非完整循环网络性能。历史数据只能在核对来源后使用。

两张新组合图会输出 PDF、可编辑文本 SVG、PNG 和来源记录，并检查最终画布的面板对齐。正式数据出图后仍需进行碰撞、文字可读性和来源核验。SI-DMS 组合图的输入要求见其原始数据说明。

最终 PDF 可在已有 PyMuPDF 的环境中运行如下检查，检查完成后还应目视确认：

```bash
python _vendor/audit_figure_collisions.py "01-机制分析/输出图片/mechanism_composite.pdf"
python _vendor/audit_figure_collisions.py "03-时间核/输出图片/temporal_kernels_composite.pdf"
python _vendor/audit_pdf_text.py "01-机制分析/输出图片/mechanism_composite.pdf" --min-pt 5
python _vendor/audit_pdf_text.py "03-时间核/输出图片/temporal_kernels_composite.pdf" --min-pt 5
```
