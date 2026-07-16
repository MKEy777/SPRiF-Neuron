"""
S-MNIST 数据加载工具（供 train / landscape / grad 共用）。

Sequential MNIST：每张 (1,28,28) 图按行展开成 (784, 1) 序列，无 permutation。
"""
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset

from config import DATA_ROOT


class SequentialMNIST(Dataset):
    """按行读取像素的 Sequential MNIST。"""

    def __init__(self, mnist: Dataset):
        self.mnist = mnist

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        img, label = self.mnist[idx]
        seq = img.reshape(-1, 1)  # (784, 1)
        return seq, label


def _base_datasets():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_mnist = torchvision.datasets.MNIST(
        root=DATA_ROOT, train=True, download=True, transform=transform)
    test_mnist = torchvision.datasets.MNIST(
        root=DATA_ROOT, train=False, download=True, transform=transform)
    return SequentialMNIST(train_mnist), SequentialMNIST(test_mnist)


def get_loaders(batch_size: int, device: torch.device, subset: int = 0):
    """返回 (train_loader, test_loader)。subset>0 时只取前 N 个训练样本用于快速验证。"""
    train_ds, test_ds = _base_datasets()
    if subset and subset > 0:
        train_ds = Subset(train_ds, list(range(min(subset, len(train_ds)))))
        test_ds = Subset(test_ds, list(range(min(subset, len(test_ds)))))

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4,
        pin_memory=pin, persistent_workers=True, prefetch_factor=2)
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=4,
        pin_memory=pin, persistent_workers=True, prefetch_factor=2)
    return train_loader, test_loader


def get_single_batch(batch_size: int, device: torch.device):
    """取一个测试 batch（梯度分析用）。"""
    _, test_ds = _base_datasets()
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    x, y = next(iter(loader))
    return x.to(device), y.to(device)