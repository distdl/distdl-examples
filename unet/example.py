from mpi4py import MPI

import os, os.path
import sys

import numpy as np
import torch

import distdl
from distdl.backends.mpi.partition import MPIPartition

from dist_unet import DistributedUNet
from logging_timer import MPILoggingTimer

from random_ellipses import gen_data

# n is total features, will be divided amongst workers
# program_name logfile.csv nf1 nf2 ... nfd pf1 pf2 ... pfd
print(sys.argv)

outfile = sys.argv[1]

# Setup logging output
timer = MPILoggingTimer()

# Parse configuration inputs
input_info = [int(v) for v in sys.argv[2:]]

feature_dimension = len(input_info) // 2
input_features = tuple(input_info[:feature_dimension])
input_workers = tuple(input_info[feature_dimension:])

n_workers = np.prod(input_workers)

# Setup some partitions:
# 1) P_world: all possible workers
# 2) P_base: enough workers to satisfy n_workers
# 3) P_0/P_root: A partition to create data on (until we create it in parallel)
# 4) P_unet: partition given by input_workers of appropriate dimension

P_world = MPIPartition(MPI.COMM_WORLD)
P_base = P_world.create_partition_inclusive(np.arange(n_workers))

# 2+feature_dimension comes batch x channel x f0 x f1 x ... fd
P_root_shape = [1]*(2+feature_dimension)
P_0 = P_base.create_partition_inclusive([0])
P_root = P_0.create_cartesian_topology_partition(P_root_shape)

P_unet_shape = [1, 1] + list(input_workers)
P_unet = P_base.create_cartesian_topology_partition(P_unet_shape)

#################################

depth = 2
in_channels = 1
base_channels = 64
out_channels = 1

unet = DistributedUNet(P_root, P_unet, depth, in_channels, base_channels, out_channels)

#################################

n_batch = 1
batch_size = 1

from distdl.utilities.tensor_decomposition import compute_subtensor_shapes_balanced
from distdl.utilities.tensor_decomposition import compute_subtensor_start_indices
from distdl.utilities.tensor_decomposition import compute_subtensor_stop_indices
from distdl.utilities.torch import TensorStructure

global_input_tensor_structure = TensorStructure()
global_input_tensor_structure.shape = input_features
subtensor_shapes = compute_subtensor_shapes_balanced(global_input_tensor_structure, P_unet.shape[2:])
subtensor_starts = compute_subtensor_start_indices(subtensor_shapes)
subtensor_stops = compute_subtensor_stop_indices(subtensor_shapes)
_slice = tuple([slice(i, i+1) for i in P_unet.index[2:]])
my_start = subtensor_starts[_slice].squeeze()
my_stop = subtensor_stops[_slice].squeeze()

MPI.COMM_WORLD.Barrier()

t_sample_spacing = [np.linspace(0, 1, f) for f in input_features]
sample_spacing = []
for d in range(len(input_features)):
    sample_spacing.append(t_sample_spacing[d][my_start[d]:my_stop[d]])
sample_grid = np.meshgrid(*sample_spacing)

# Ellipses are created from random parameters, each rank will generate the
# same sequence of parameters, so each ellipse can be evaluated in parallel
np.random.seed(0)

n_ellipses_target = 3
n_ellipses_noise = 2

timer.start("data gen")
batches = list()
for i in range(n_batch):
    batch = list()
    for j in range(batch_size):
        # Add an image-mask tuple to the batch
        batch.append(gen_data(sample_grid, n_ellipses_target, n_ellipses_noise))
    img = torch.cat([im for im, ma in batch],dim=0)
    mask = torch.cat([ma for im, ma in batch],dim=0)
    batches.append((img, mask))
timer.stop("data gen", input_features)

# Leave the demo where we generate the code in one rank and scatter it...for posterity
#
# scatter = distdl.nn.DistributedTranspose(P_root, P_unet)
#
# with torch.no_grad():
#     if P_root.active:

#         sample_spacing = [np.linspace(0, 1, f) for f in input_features]
#         sample_grid = np.meshgrid(*sample_spacing)

#         n_ellipses_target = 3
#         n_ellipses_noise = 2

#         timer.start("data gen")
#         batches = list()
#         for i in range(n_batch):
#             batch = list()
#             for j in range(batch_size):
#                 # Add an image-mask tuple to the batch
#                 batch.append(gen_data(sample_grid, n_ellipses_target, n_ellipses_noise))
#             img = torch.cat([im for im, ma in batch],dim=0)
#             img = scatter(img)
#             mask = torch.cat([ma for im, ma in batch],dim=0)
#             mask = scatter(mask)

#             batches.append((img, mask))
#         timer.stop("data gen", input_features)
#     else:
#         timer.start("data gen")
#         batches = list()
#         for i in range(n_batch):
#             batch = list()
#             for j in range(batch_size):
#                 img = distdl.utilities.torch.zero_volume_tensor(batch_size)
#                 img = scatter(img)
#                 mask = distdl.utilities.torch.zero_volume_tensor(batch_size)
#                 mask = scatter(mask)

#             batches.append((img, mask))
#         timer.stop("data gen", input_features)

MPI.COMM_WORLD.Barrier()

#################################

n_epoch = 5

parameters = [p for p in unet.parameters()]

# Hack to make empty parts of the graph happy
if not parameters:
    parameters = [torch.nn.Parameter(torch.zeros(1))]

optimizer = torch.optim.Adam(parameters,lr=0.0001)
criterion = distdl.nn.DistributedBCEWithLogitsLoss(P_unet)

################################

for j in range(n_epoch):

    MPI.COMM_WORLD.Barrier()

    timer.start("epoch")

    for i in range(n_batch):

        MPI.COMM_WORLD.Barrier()

        timer.start("batch")
        img, mask = batches[i]

        optimizer.zero_grad()

        timer.start("forward")
        out = unet(img)
        timer.stop("forward", f"{j}, {i}")

        timer.start("loss")
        loss = criterion(out, mask)
        loss_value = loss.item()
        timer.stop("loss", f"{j}, {i}")

        if P_root.active:
            print(f"Loss: {loss_value}")

        timer.start("adjoint")
        loss.backward()
        timer.stop("adjoint", f"{j}, {i}")

        timer.start("step")
        optimizer.step()
        timer.stop("step", f"{j}, {i}")

        timer.stop("batch", f"{j}, {i}")

        if P_root.active:
            timer.to_csv(outfile)

    timer.stop("epoch", j)

if P_root.active:
    timer.to_csv(outfile)
