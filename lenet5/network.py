import numpy as np
import torch
from layers import DistributedNetworkOutput
from layers import Flatten
from mpi4py import MPI

import distdl


def generate_sequential_network():

    net = torch.nn.Sequential(torch.nn.Conv2d(1, 6,
                                              kernel_size=(5, 5),
                                              padding=(2, 2)),
                              torch.nn.ReLU(),
                              torch.nn.MaxPool2d(kernel_size=(2, 2),
                                                 stride=(2, 2)),
                              torch.nn.Conv2d(6, 16,
                                              kernel_size=(5, 5),
                                              padding=(0, 0)),
                              torch.nn.ReLU(),
                              torch.nn.MaxPool2d(kernel_size=(2, 2),
                                                 stride=(2, 2)),
                              torch.nn.Conv2d(16, 120,
                                              kernel_size=(5, 5),
                                              padding=(0, 0)),
                              torch.nn.ReLU(),
                              Flatten(),
                              torch.nn.Linear(120, 84),
                              torch.nn.ReLU(),
                              torch.nn.Linear(84, 10),
                              torch.nn.Sigmoid()
                              )

    return net

def generate_distributed_network():

    P_world = distdl.backend.backend.Partition(MPI.COMM_WORLD)
    P_world._comm.Barrier()

    # Base partition the network is ditributed across
    P_base = P_world.create_partition_inclusive(np.arange(4))

    # Partition used for input/output
    P_0 = P_base.create_partition_inclusive([0])
    P_root = P_0.create_cartesian_topology_partition([1, 1, 1, 1])
    P_root_2d = P_0.create_cartesian_topology_partition([1, 1])

    # Disjoint partitions of the base used for fully connected layer input/output
    P_base_lo = P_base.create_partition_inclusive(np.arange(0, 2))
    P_base_hi = P_base.create_partition_inclusive(np.arange(2, 4))

    # Cartesian partitions needed for decompositon of layers
    P_conv = P_base.create_cartesian_topology_partition([1, 1, 2, 2])
    P_flat = P_base.create_cartesian_topology_partition([1, 4, 1, 1])
    P_flat_2d = P_base.create_cartesian_topology_partition([1, 4])
    P_fc_in = P_base_lo.create_cartesian_topology_partition([1, 2])
    P_fc_out = P_base_hi.create_cartesian_topology_partition([1, 2])
    P_fc_mtx = P_base.create_cartesian_topology_partition([2, 2])

    # net = distdl.nn.Distributed(distdl.nn.DistributedTranspose(P_root,
    net = torch.nn.Sequential(distdl.nn.DistributedTranspose(P_root,
                                                             P_conv),
                              distdl.nn.DistributedConv2d(P_conv,
                                                          in_channels=1,
                                                          out_channels=6,
                                                          kernel_size=(5, 5),
                                                          padding=(2, 2)),
                              torch.nn.ReLU(),
                              distdl.nn.DistributedMaxPool2d(P_conv,
                                                             kernel_size=(2, 2),
                                                             stride=(2, 2)),
                              distdl.nn.DistributedConv2d(P_conv,
                                                          in_channels=6,
                                                          out_channels=16,
                                                          kernel_size=(5, 5),
                                                          padding=(0, 0)),
                              torch.nn.ReLU(),
                              distdl.nn.DistributedMaxPool2d(P_conv,
                                                             kernel_size=(2, 2),
                                                             stride=(2, 2)),
                              distdl.nn.DistributedTranspose(P_conv,
                                                             P_flat),
                              Flatten(),
                              distdl.nn.DistributedTranspose(P_flat_2d,
                                                             P_fc_in),
                              distdl.nn.DistributedLinear(P_fc_in,
                                                          P_fc_out,
                                                          P_fc_mtx,
                                                          400, 120),
                              torch.nn.ReLU(),
                              # Reverse order of P_fc_in and P_fc_out to avoid a transpose
                              distdl.nn.DistributedLinear(P_fc_out,
                                                          P_fc_in,
                                                          P_fc_mtx,
                                                          120, 84),
                              torch.nn.ReLU(),
                              distdl.nn.DistributedLinear(P_fc_in,
                                                          P_fc_out,
                                                          P_fc_mtx,
                                                          84, 10),
                              distdl.nn.DistributedTranspose(P_fc_out,
                                                             P_root_2d),
                              torch.nn.Sigmoid(),
                              DistributedNetworkOutput(P_root_2d)
                              )

    return P_base, net
