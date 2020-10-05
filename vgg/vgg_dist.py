# Adapted from https://github.com/pytorch/vision/blob/v0.5.1/torchvision/models/vgg.py under
# the Torchvision license, reproduced below.  All modifications follow the standard DistDL
# license.

# BSD 3-Clause License

# Copyright (c) Soumith Chintala 2016, All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import torch
import torch.nn as nn
import torchvision
from layers import DistributedNetworkOutput
from layers import Flatten

import distdl


class DistributedVGG(distdl.nn.Module):

    def __init__(self, vgg_cfg_str, P_world,
                 num_classes=1000, init_weights=True):
        super(DistributedVGG, self).__init__()

        P_world._comm.Barrier()

        # Find the length of the side for a square partition (conv, pool)
        # This is also the size of the linear partitions, as the weight
        # matrix is partitioned the same way
        p = np.int(np.sqrt(P_world.size))

        # Find the total size of the square partitions
        p2 = p*p

        # Base partition the network is ditributed across
        P_base = P_world.create_partition_inclusive(np.arange(p2))
        self.P_base = P_base

        # Partition used for input/output
        P_0 = P_base.create_partition_inclusive([0])
        P_root = P_0.create_cartesian_topology_partition([1, 1, 1, 1])
        P_root_2d = P_0.create_cartesian_topology_partition([1, 1])

        # Disjoint partitions of the base used for fully connected layer input/output
        P_base_lo = P_base.create_partition_inclusive(np.arange(0, p))
        P_base_hi = P_base.create_partition_inclusive(np.arange(p, 2*p))

        # Cartesian partitions needed for decompositon of layers
        P_feat = P_base.create_cartesian_topology_partition([1, 1, p, p])
        P_feat_flat = P_base.create_cartesian_topology_partition([1, p2, 1, 1])
        P_feat_flat_2d = P_base.create_cartesian_topology_partition([1, p2])
        P_class_in = P_base_lo.create_cartesian_topology_partition([1, p])
        P_class_out = P_base_hi.create_cartesian_topology_partition([1, p])
        P_class_mtx = P_base.create_cartesian_topology_partition([p, p])

        # Create some dummy layers, not really needed.
        if not P_base.active:
            self.input_map = lambda x: x
            self.features = lambda x: x
            self.feat_to_class_map = lambda x: x
            self.classifier = lambda x: x
            self.output = lambda x: x
            return

        # Maps input from one worker to the feature workers
        self.input_map = distdl.nn.DistributedTranspose(P_root, P_feat)

        # Create the feature mapping portion of the network
        cfg = torchvision.models.vgg.cfg[vgg_cfg_str]
        self.features = make_layers(cfg, P_feat)

        # In the Torchvision implementation, this layer handles oddly sized
        # input and ensures that the input to the classifier has correct size.
        # Currently, we resize all inputs so that they are 224x224, so the
        # input size is always correct.  We don't need this layer.
        #
        # self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        # Some minor transformations are required to go from the feature mapping
        # to the classifier.
        self.feat_to_class_map = torch.nn.Sequential(
            distdl.nn.DistributedTranspose(P_feat,
                                           P_feat_flat),
            Flatten(),
            distdl.nn.DistributedTranspose(P_feat_flat_2d,
                                           P_class_in),
        )

        # Distributed form of the VGG classifier.
        self.classifier = torch.nn.Sequential(
            distdl.nn.DistributedLinear(P_class_in, P_class_out, P_class_mtx, 512 * 7 * 7, 4096),
            torch.nn.ReLU(True),
            torch.nn.Dropout(),
            distdl.nn.DistributedLinear(P_class_out, P_class_in, P_class_mtx, 4096, 4096),
            torch.nn.ReLU(True),
            torch.nn.Dropout(),
            distdl.nn.DistributedLinear(P_class_in, P_class_out, P_class_mtx, 4096, num_classes),
            distdl.nn.DistributedTranspose(P_class_out, P_root_2d)
        )

        # One more trick is needed to have an output that the optimizers can
        # work with in parallel.
        self.output = DistributedNetworkOutput(P_root_2d)

        if init_weights:
            self._initialize_weights()

    def forward(self, x):

        x = self.input_map(x)

        x = self.features(x)

        # See above.
        # x = self.avgpool(x)

        x = self.feat_to_class_map(x)

        x = self.classifier(x)

        x = self.output(x)

        return x

    def _initialize_weights(self):

        for m in self.modules():
            if isinstance(m, distdl.nn.DistributedFeatureConv2d):
                # For this layer, only the active workers in this partition
                # have the weight function and bias
                if m.P_wb_cart.active:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.conv_layer.bias is not None:
                        nn.init.constant_(m.bias, 0)
            # BatchNorm is not supported, at the moment.
            # elif isinstance(m, distdl.nn.DistributedBatchNorm):
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0)
            elif isinstance(m, distdl.nn.DistributedLinear):
                if m.P_w.active:
                    nn.init.normal_(m.sublinear.weight, 0, 0.01)
                    if m.sublinear.bias is not None:
                        nn.init.constant_(m.sublinear.bias, 0)


def make_layers(cfg, P_feat, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [distdl.nn.DistributedMaxPool2d(P_feat,
                                                      kernel_size=(2, 2),
                                                      stride=(2, 2))]
        else:
            conv2d = distdl.nn.DistributedConv2d(P_feat,
                                                 in_channels=in_channels,
                                                 out_channels=v,
                                                 kernel_size=(3, 3),
                                                 padding=(1, 1))
            if batch_norm:
                # BatchNorm is not supported, at the moment.
                raise ValueError("Distributed Batch Normed implementation of VGG is not currently supported.")
                # layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)
