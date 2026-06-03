import torch
from torch import Tensor
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from config import BATCH_SIZE

class AudioDataset(Dataset):
    def __init__(self, X, y) -> None:
        self.X: Tensor = X if isinstance(X, Tensor) else torch.tensor(X, dtype=torch.float32)
        self.y: Tensor = y.long() if isinstance(y, Tensor) else torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        # (H, W, C) → (C, H, W) for Conv2d compatibility
        return self.X[idx].permute(2, 0, 1), self.y[idx]


class SyntheticDataset(Dataset):
    def __init__(self, X: Tensor) -> None:
        self.X = X
        self.y = torch.ones(len(X), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def make_augmented_loader(
    real_dataset: AudioDataset,
    fake_X: Tensor,
    batch_size: int = BATCH_SIZE,
) -> DataLoader:
    combined = ConcatDataset([real_dataset, SyntheticDataset(fake_X)])
    return DataLoader(combined, batch_size=batch_size, shuffle=True)