from torch.utils.data import DataLoader
import torch
from torch import nn
from torch.optim import Optimizer

class ForexTrainer:
    def __init__(self, model: nn.Module, optimizer: Optimizer, loss_fn: nn.Module, device: torch.device):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.train_loss_history = []
        self.val_loss_history   = []
    
    
    def train(self, train_loader: DataLoader, num_epochs: int) -> None:
        # put model in train mode
        self.model.train()
        # loop through epochs
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for batch in train_loader:
                # load
                inp = batch['input'].to(self.device)        # inputs
                tgt = batch['output'].to(self.device)       # targets
                # initialize gradients
                self.optimizer.zero_grad()
                # pass through model
                out = self.model(inp)                       # outputs
                # compute loss
                loss = self.loss_fn(out, tgt)               # compare outputs to targets
                # compute gradients
                loss.backward()
                # update weights
                self.optimizer.step()
                # update loss
                epoch_loss += loss.item()
                
            avg_loss = epoch_loss / len(train_loader)
            self.train_loss_history.append(avg_loss)
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_loss:.4f}")
        
        # python object like nn>module are mutated in place, so no need to return model
    
    
    def evaluate(self, val_loader: DataLoader) -> float:
        # put model in eval mode
        self.model.eval()
        # ignore gradients
        with torch.no_grad():
            eval_loss = 0.0
            for batch in val_loader:
                inp = batch['input'].to(self.device)
                tgt = batch['output'].to(self.device)
                
                out = self.model(inp)
                loss = self.loss_fn(tgt, out)
                eval_loss += loss.item()
                
        avg_loss = eval_loss / len(val_loader)
        self.val_loss_history.append(avg_loss)
        print(f"Validation Loss: {avg_loss:.4f}")
        return avg_loss
        
                
    def predict(self, inp: torch.Tensor) -> torch.Tensor:
        # put model in eval mode
        self.model.eval()
        # ignore gradients
        with torch.no_grad():
            # move to device
            inp = inp.to(self.device)
            # pass through model
            out = self.model(inp)
        return out
     
        
    def save_model(self, path:str) -> None:
        torch.save(self.model.state_dict(), path)
     
        
    def load_model(self, path:str) -> None:
        self.model.load_state_dict(torch.load(path))