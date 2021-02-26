import torch.nn as nn

class MyUDF(nn.Module):
    def __init__(self):
        super(MyUDF, self).__init__()
        self.calls = 0

    def forward(self, x):
        return x

__all__ = ["MyUDF"]
