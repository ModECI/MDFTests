import torch.nn


class SimpleLoopUDF(torch.nn.Module):
    def forward(self, in_num, addend=1):

        for i in range(10):
            in_num += addend

        return in_num

