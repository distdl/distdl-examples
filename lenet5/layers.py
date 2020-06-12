import torch

from distdl.utilities.torch import NoneTensor


class DistributedNetworkOutputFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, partition):

        ctx.partition = partition

        if partition.rank == 0:
            return input.clone()

        else:
            return torch.tensor([0.0], requires_grad=True).float()

    @staticmethod
    def backward(ctx, grad_output):

        partition = ctx.partition

        if partition.rank == 0:
            return grad_output.clone(), None

        else:
            return NoneTensor(), None


class DistributedNetworkOutput(torch.nn.Module):

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
