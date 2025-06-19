from dataclasses import dataclass
import torch

OPTIMIZERS = {
  "Adam": torch.optim.Adam,
  "SGD":  torch.optim.SGD,
}

@dataclass
class Config:
    # device
    device: torch.device = torch.device('cpu')
    
    # data
    input_length: int = 30
    output_length: int = 1
    target_feature_idx: int = 3
    batch_size: int = 32
    
    # architecture
    d_model: int = 64
    nhead: int = 4
    dim_feedforward: int = 32
    num_layers: int = 3
    
    # optimizer
    optimizer_type: str = "Adam"
    lr: float = 1e-3
    weight_decay: float = 0
    
    # loss
    loss_fn: torch.nn.Module = torch.nn.MSELoss()
    
    # training
    num_epochs: int = 20