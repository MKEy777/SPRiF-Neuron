# 项目背景笔记 — Health Check 命令

## 项目概述

SPRiF (Slow-Path Resonance with Intrinsic Fast-flow) 是一个脉冲神经网络 (SNN) 研究项目。
使用 PyTorch 实现，包含多个基准任务（GSC、SHD、ECG、S-MNIST、pSMNIST）。
无包管理器配置文件（如 package.json），纯 Python 项目，使用 `requirements.txt` 管理依赖。

## 项目结构

```
代码/                          # 主体代码
├── Task_GSC/                 # Google Speech Commands (12 类关键词识别)
│   ├── train.py              # 训练入口 (argparse 解析参数)
│   ├── model.py              # 网络定义 (SPRiFGSCNet)
│   ├── data.py               # 数据加载 (SpeechCommandsDataset)
│   ├── utils.py              # 工具函数
│   ├── core_algorithm/       # 共享算法库 (各任务独立拷贝)
│   │   ├── __init__.py
│   │   ├── sprif_layer.py    # SPRiFNeuronLayer + surrogate gradient
│   │   └── utils.py          # set_seed, dump_json, load_json 等
│   └── train_ablation_{a,b,c,d}.py
├── Task_SHD/                 # Spiking Heidelberg Digits (20 类)
│   ├── train.py / model.py / data.py / generate_data.py
│   └── core_algorithm/       (同上)
├── Task_ECG/                 # QTDB ECG 分类 (6 类)
│   ├── train.py / model.py
│   └── core_algorithm/       (同上)
├── Task_S-MNIST/             # Sequential MNIST (10 类)
│   ├── train.py / model.py
│   └── core_algorithm/       (同上)
├── Task_pSMNIST/             # Permuted Sequential MNIST (10 类)
│   ├── train.py / model.py / data/
│   └── core_algorithm/       (同上)
├── experiments/              # 分析脚本
│   ├── trajectory_analysis/
│   ├── impulse_analysis/
│   ├── reset_analysis/
│   ├── param_visualization/
│   └── robustness/
└── requirements.txt          # 依赖列表

ASRNN/                        # ASRNN 基线代码
├── GSC/
│   ├── train_asrnn.py / srnn_fin.py / optim.py / data.py / utils.py
│   └── SRNN_layers/ / model_wrapper/
├── ECG/
└── data/
```

## 现有模式

- 所有任务入口统一使用 `argparse.ArgumentParser`
- 训练脚本可独立运行：`python train.py [args]`
- 评估函数均为 `@torch.no_grad()` 装饰的只读模式
- 分析脚本中有通用的 `_find_checkpoint()` 工具函数
- core_algorithm 在各任务中独立拷贝（需检查一致性）

## 需求：只读健康检查命令

预期功能：
1. 检查 Python / PyTorch / CUDA 环境
2. 检查项目目录结构完整性
3. 检查各任务数据集是否存在
4. 检查已训练的 checkpoint 文件
5. 检查 core_algorithm 一致性（各任务是否同步）
6. 检查依赖是否安装
7. 汇总报告（stdout），不写任何文件
