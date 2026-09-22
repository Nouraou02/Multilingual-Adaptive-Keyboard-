import torch.nn as nn
class Router(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )
    
    def forward(self,x):
        return self.mlp(x)
    