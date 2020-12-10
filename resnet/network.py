from mpi4py import MPI
from torchvision.models import resnet18
from resnet_dist import resnet18_dist

import distdl


def generate_sequential_network(num_classes):

    return resnet18(num_classes=num_classes)

def generate_distributed_network(num_classes):

    P_world = distdl.backend.backend.Partition(MPI.COMM_WORLD)
    P_world._comm.Barrier()

    return resnet18_dist(P_world, parts=[4, 8], num_classes=num_classes)
