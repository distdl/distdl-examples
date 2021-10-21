import torch
import torch.nn
import distdl

class Concatenate(torch.nn.Module):

    def __init__(self, axis):
        super(Concatenate, self).__init__()

        self.axis = axis

    def forward(self, *args):

        return torch.cat(*args, self.axis)


# If this is true, Autograd does not like the inplaceness of the halo exchage
# My gut feeling is that the halo exchange is more expensive memory-wise
# than ReLU, so I prefer to keep the halo exchange as inplace.
# https://github.com/distdl/distdl/issues/199
_relu_inplace = False

class DistributedUNet(torch.nn.Module):

    def __init__(self, P_root, P, levels, in_channels, base_channels, out_channels):
        super(DistributedUNet, self).__init__()

        self.levels = levels

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.out_channels = out_channels

        self.P_root = P_root
        self.P = P
        self.ConvType = distdl.nn.DistributedConv3d
        self.PoolType = distdl.nn.DistributedMaxPool3d

        self.input_map = self.assemble_input_map()
        self.unet = self.assemble_unet()
        self.output_map = self.assemble_output_map()


    def assemble_input_map(self):

        conv = self.ConvType(self.P,
                             in_channels=self.in_channels,
                             out_channels=self.base_channels,
                             kernel_size=3, padding=1)
        norm = distdl.nn.DistributedBatchNorm(self.P,
                                              num_features=self.base_channels)
        acti = torch.nn.ReLU(inplace=_relu_inplace)
        return torch.nn.Sequential(conv, norm, acti)


    def assemble_unet(self):
        return DistributedUNetLevel(self.P, self.levels, 0, self.base_channels)

    def assemble_output_map(self):

        conv =self.ConvType(self.P,
                            in_channels=self.base_channels,
                            out_channels=self.out_channels,
                            kernel_size=1)
        # Original study does not have these.  Also note kernel_size=1 above.
        # norm = distdl.nn.DistributedBatchNorm(self.P,
        #                                       num_features=self.out_channels)
        # acti = torch.nn.ReLU(inplace=_relu_inplace)
        # out = DistributedNetworkOutput(self.P)
        return torch.nn.Sequential(conv)  #, norm, acti)


    def forward(self, input):

        x_f = self.input_map(input)
        y_f = self.unet(x_f)
        output = self.output_map(y_f)

        return output


class DistributedUNetLevel(torch.nn.Module):

    def __init__(self, P, max_levels, level, base_channels):

        super(DistributedUNetLevel, self).__init__()

        self.max_levels = max_levels
        self.level = level

        self.base_channels = base_channels

        self.coarsest = (self.level == self.max_levels-1)

        self.P = P
        self.ConvType = distdl.nn.DistributedConv3d
        self.PoolType = distdl.nn.DistributedMaxPool3d

        if self.coarsest:
            self.coarse_filter = self.assemble_coarse_filter()
        else:
            self.pre_filter = self.assemble_pre_filter()
            self.downscale = self.assemble_downscale()
            self.sublevel = self.assemble_sublevel()
            self.upscale = self.assemble_upscale()
            self.correction = self.assemble_correction()
            self.post_filter = self.assemble_post_filter()


    def channels(self, level=None):
        if level is None:
            level = self.level
        return (2**level)*self.base_channels


    def assemble_coarse_filter(self):

        channels = self.channels()
        conv = self.ConvType(self.P,
                             in_channels=channels,
                             out_channels=channels,
                             kernel_size=3, padding=1)
        norm = distdl.nn.DistributedBatchNorm(self.P,
                                              num_features=channels)
        acti = torch.nn.ReLU(inplace=_relu_inplace)

        return torch.nn.Sequential(conv, norm, acti)


    def assemble_pre_filter(self):

        channels = self.channels()
        conv = self.ConvType(self.P,
                             in_channels=channels,
                             out_channels=channels,
                             kernel_size=3, padding=1)
        norm = distdl.nn.DistributedBatchNorm(self.P,
                                              num_features=channels)
        acti = torch.nn.ReLU(inplace=_relu_inplace)

        return torch.nn.Sequential(conv, norm, acti)


    def assemble_post_filter(self):

        channels = self.channels()
        conv = self.ConvType(self.P,
                             in_channels=channels,
                             out_channels=channels,
                             kernel_size=3, padding=1)
        norm = distdl.nn.DistributedBatchNorm(self.P,
                                              num_features=channels)
        acti = torch.nn.ReLU(inplace=_relu_inplace)

        return torch.nn.Sequential(conv, norm, acti)


    def assemble_downscale(self):

        in_channels = self.channels()
        out_channels = self.channels(self.level+1)

        pool = self.PoolType(self.P, kernel_size=2, stride=2)
        conv = self.ConvType(self.P,
                             in_channels=in_channels,
                             out_channels=out_channels,
                             kernel_size=3, padding=1)
        norm = distdl.nn.DistributedBatchNorm(self.P,
                                              num_features=out_channels)
        acti = torch.nn.ReLU(inplace=_relu_inplace)
        return torch.nn.Sequential(pool, conv, norm, acti)


    def assemble_upscale(self):

        in_channels = self.channels(self.level+1)
        out_channels = self.channels(self.level)

        up = distdl.nn.DistributedUpsample(self.P,
                                           scale_factor=2)
        conv = self.ConvType(self.P,
                             in_channels=in_channels,
                             out_channels=out_channels,
                             kernel_size=1)
        # Original study does not have these.  Also note kernel_size=1 above.
        # norm = distdl.nn.DistributedBatchNorm(self.P,
        #                                       num_features=out_channels)
        # acti = torch.nn.ReLU(inplace=_relu_inplace)
        return torch.nn.Sequential(up, conv)  #, norm, acti)


    def assemble_correction(self):

        in_channels = 2*self.channels()
        out_channels = self.channels()

        add = Concatenate(1)
        conv = self.ConvType(self.P,
                             in_channels=in_channels,
                             out_channels=out_channels,
                             kernel_size=3, padding=1)
        norm = distdl.nn.DistributedBatchNorm(self.P,
                                              num_features=out_channels)
        acti = torch.nn.ReLU(inplace=_relu_inplace)
        return torch.nn.Sequential(add, conv, norm, acti)


    def instantiate_sublevel(self):
        return DistributedUNetLevel(self.P, self.max_levels, self.level+1, self.base_channels)


    def assemble_sublevel(self):

        # If this level is less than one less than the max, it is a coarsest level
        if not self.coarsest:
            return self.instantiate_sublevel()
        else:
            raise Exception()

    def forward(self, x_f):

        if self.coarsest:
            y_f = self.coarse_filter(x_f)
            return y_f

        y_f = self.pre_filter(x_f)
        y_c = self.downscale(y_f)

        y_c = self.sublevel(y_c)

        y_c = self.upscale(y_c)
        y_f = self.correction((y_f, y_c))
        y_f = self.post_filter(y_f)

        return y_f
