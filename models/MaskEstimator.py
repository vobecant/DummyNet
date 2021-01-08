import sys, os

from models.Losses import KL_loss
from models.customLayers import UNetDown, UNetUp, UNetCore

tmp = '/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-1])
sys.path.append(tmp)

from torch.nn import ModuleList, Sigmoid, Conv2d
from torch import nn


class LinearActivation(nn.Module):
    def __init__(self):
        super(LinearActivation, self).__init__()

    def forward(self, x):
        return x


class MaskEstimator(nn.Module):
    def __init__(self, args, input_channels):
        super(MaskEstimator, self).__init__()

        self.blocks_down = ModuleList([])
        self.blocks_up = ModuleList([])
        self.to_mask = None
        self.input_channels = input_channels

        c_out = self.input_channels

        for i in range(1, args.depth + 1):
            c_in = int(c_out)
            c_out = c_in * 2  # if i > 1 else 16  # more output than input channels
            down = UNetDown(c_in, c_out, args.use_eql)
            up = UNetUp(c_out, c_in, args.use_eql)
            if i == 1:
                self.to_mask = Conv2d(c_out, 1, 3, 1, 1)
            self.blocks_down.append(down)
            self.blocks_up.append(up)

        self.bottleneck = UNetCore(c_out, 2 * c_out, use_eql=args.use_eql)

        try:
            activation = args.output_activation.lower()
            if activation == 'sigmoid':
                self.output_activation = Sigmoid()
            elif activation == 'linear' or activation == 'none':
                self.output_activation = LinearActivation()
            else:
                raise Exception('Unknown output activation!')
        except:
            self.output_activation = Sigmoid()

    def forward(self, keypoints, return_background=True):
        # all layers of stream down
        y = keypoints
        residuals = []
        for block_down in self.blocks_down:
            y, residual = block_down(y)
            residuals.append(residual)
        y = self.bottleneck(y)
        for block_up, residual in zip(reversed(self.blocks_up),
                                      reversed(residuals)):
            y = block_up(y, residual)
        estimated_mask = self.to_mask(y)
        estimated_mask = self.output_activation(estimated_mask)
        background = 1 - estimated_mask
        background[background > 0.1] = 1

        if return_background:
            # return out, new_pixels, wp
            return estimated_mask, background
        else:
            return estimated_mask
