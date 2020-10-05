from mpi4py import MPI
from torchvision.models import vgg16
from vgg_dist import DistributedVGG

import distdl


def generate_sequential_network(num_classes):

    return vgg16(num_classes=num_classes)

def generate_distributed_network(num_classes):

    P_world = distdl.backend.backend.Partition(MPI.COMM_WORLD)
    P_world._comm.Barrier()

    # VGG structure 'D'
    return DistributedVGG("D", P_world, num_classes=num_classes)
