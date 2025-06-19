from torch.utils.data import Dataset
import numpy as np
import torch

class ForexDataset(Dataset):
    # must contain __init__(self, ...), __len__(self), __getitem__(self, idx)
    # getitem must return tuple or dict
    
    def __init__(self, data: np.ndarray, input_length: int, output_length: int, target_feature_idx: int) -> None:
        self.data = torch.tensor(data, dtype=torch.float32)                # [num_samples, num_features]
        self.input_length = input_length
        self.output_length = output_length
        self.target_feature_idx = target_feature_idx
    
    def __len__(self) -> int:
        return len(self.data) - self.input_length - self.output_length + 1  # len returns size of first dimension
    
    def __getitem__(self, idx) -> dict[str, torch.Tensor]:
        x = self.data[idx : idx + self.input_length, :]
        y = self.data[idx + self.input_length : idx + self.input_length + self.output_length, self.target_feature_idx]
        return {'input': x, 'output': y}