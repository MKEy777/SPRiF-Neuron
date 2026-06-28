# Training Logs

记录每个实验的训练曲线和超参数，供论文 Appendix 使用。

## 期望内容

### 训练曲线

每个实验保存一个 CSV，包含每 epoch 的 train_loss / test_acc：

```
{experiment_name}_curve.csv

列: epoch, train_loss, test_accuracy, learning_rate
```

示例: `SPRiF_GSC_seed42_curve.csv`, `ablation_A_PSMNIST_seed0_curve.csv`

### 超参数汇总

一个汇总 JSON，记录所有实验的最终超参数：

```json
{
  "experiment": "SPRiF_GSC_seed42",
  "lr": 0.003,
  "epochs": 150,
  "batch_size": 200,
  "optimizer": "AdamW",
  "weight_decay": 0.0001,
  "scheduler": "StepLR(step=30, gamma=0.5)",
  "grad_clip": 10.0,
  "neuron_threshold": 1.0,
  "hidden_sizes": [300],
  "best_epoch": 127,
  "best_test_acc": 0.9512,
  "total_params": 123456,
  "train_time_hours": 2.3,
  "gpu": "RTX 4090"
}
```

### Checkpoint 路径

记录每个实验的 .pth 文件路径，方便复现：

```
SPRiF_GSC_seed42        → Task_GSC/SPRiFGSCNet_hs[300]_bs200_lr0.003_seed42_acc0.9512.pth
SPRiF_PSMNIST_seed0     → Task_pSMNIST/SPRiFpSMNISTNet_hs[64,256]_bs512_lr0.01_seed0_acc0.9734.pth
...
```

---

## 无需手动填写

这些文件可以在实验跑完后由训练脚本自动生成，或从实验输出中提取。
如果你不想手动整理训练日志，只需提供 checkpoint 文件路径，我可以从 .pth 中提取参数统计。
