import os
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

class SHDDataset(Dataset):
    def __init__(self, data_paths: List[str], transform: Optional[callable] = None):
        self.data_paths = data_paths
        self.transform = transform

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        x = torch.from_numpy(np.load(self.data_paths[index])).float()
        y_str = os.path.basename(self.data_paths[index]).split("_")[-1]
        y = int(y_str.split(".")[0])
        y = torch.tensor(y, dtype=torch.long)
        if self.transform is not None:
            x = self.transform(x)
        return x, y

