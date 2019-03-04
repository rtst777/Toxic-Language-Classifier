import torch
import torch.nn as nn
import random

# only used for testing the integration with the server
class DummyNet(nn.Module):
    def __init__(self):
        super(DummyNet, self).__init__()
        self.name = "DummyNet"

    def forward(self, x):
        rand1 = random.randint(1,101) / 100
        rand2 = random.randint(1, 101) / 100
        rand3 = random.randint(1, 101) / 100
        return torch.tensor([rand1, rand2, rand3])
