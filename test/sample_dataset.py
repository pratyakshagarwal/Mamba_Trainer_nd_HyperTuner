import torch
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, data: list[int], block_size: int):
        self.data = data
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(self.data[idx:idx + self.block_size], dtype=torch.int64)
        y = torch.tensor(self.data[idx + 1:idx + self.block_size + 1], dtype=torch.int64)
        return x, y