import torch

import distdl
from distdl.utilities.torch import zero_volume_tensor


class DistributedNetworkOutputFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, partition):

        ctx.partition = partition
        ctx.sh = input.shape

        if partition.rank == 0:
            return input.clone()
        else:
            return torch.tensor([0.0], requires_grad=True).float()

    @staticmethod
    def backward(ctx, grad_output):

        partition = ctx.partition
        sh = ctx.sh

        if partition.rank == 0:
            return grad_output.clone(), None
        else:
            return zero_volume_tensor(sh[0]), None


class DistributedNetworkOutput(distdl.nn.Module):

    def __init__(self, partition):
        super(DistributedNetworkOutput, self).__init__()
        self.partition = partition

    def forward(self, input):

        return DistributedNetworkOutputFunction.apply(input, self.partition)


class Flatten(torch.nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input):
        batch_size = input.size(0)
        return input.view(batch_size, -1)
