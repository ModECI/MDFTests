import torch.nn
import psyneulink as pnl

class SimpleLoopUDF(torch.nn.Module):
    def __init__(self):
        super(SimpleLoopUDF, self).__init__()

    def forward(self):
        return 5

"""
If using a pnl function, put the execute in a forward method
"""

__all__ = ["SimpleLoopUDF"]
