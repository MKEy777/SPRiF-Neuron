import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset

from config import DATA_ROOT

class SequentialMNIST(Dataset):

    def __init__(self, mnist: Dataset):
        self.mnist = mnist

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        img, label = self.mnist[idx]
        seq = img.reshape(-1, 1)
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

