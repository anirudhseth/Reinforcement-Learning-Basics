import torch
import torch.nn as nn
import torch.nn.functional as F

class Neural_Network(nn.Module):

    def __init__(self, input_dim, output_dim, seed):
        super(Neural_Network, self).__init__()  
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128,64)
        self.out = nn.Linear(64, output_dim)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        out = self.out(x)
        return out  