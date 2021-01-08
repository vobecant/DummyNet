import sys, os

tmp = '/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-1])
sys.path.append(tmp)
import torch as th
from torch.nn import Sigmoid, Sequential, Dropout, BatchNorm2d, AvgPool2d
from torch.nn.functional import conv_transpose2d
from torch.nn import Conv2d, ConvTranspose2d, LeakyReLU, ReLU, MaxPool2d
from models.utils import *
from torch import nn
import torch.nn.functional as F


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    """
    Based on the paper "Spectral Normalization for Generative Adversarial Networks" by Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida
    and the Pytorch implementation https://github.com/christiancosgrove/pytorch-spectral-normalization-gan
    """

    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


###
### CONVOLUTIONAL LAYERS
###

# extending Conv2D and Deconv2D layers for equalized learning rate logic
class _equalized_conv2d(th.nn.Module):
    """ conv2d with the concept of equalized learning rate
    EXPLANATION FROM PAPER:
    They explicitly scale the weights at runtime.
    They set the weight w_i = w_i/c where 'w_i' are weights and 'c' is the per-layer normalization constant
    from He's initializer.
    As a result, if some parameters have a larger dynamic range than others, they will take longer to adjust.
    """

    def __init__(self, c_in, c_out, k_size, stride=1, pad=0, bias=True, dilation=False):
        """ conv2d with the concept of equalized learning rate
            Args:
                :param c_in: input channels
                :param c_out:  output channels
                :param k_size: kernel size (h, w) should be a tuple or a single integer
                :param stride: stride for conv
                :param pad: padding
                :param bias: whether to use bias or not
        """
        """ constructor for the class """
        from torch.nn.modules.utils import _pair
        from numpy import sqrt, prod

        super(_equalized_conv2d, self).__init__()

        # define the weight and bias if to be used with Normal distribution (0,1)
        self.weight = th.nn.Parameter(th.nn.init.normal_(
            th.empty(c_out, c_in, *_pair(k_size))
        ))

        self.use_bias = bias
        self.stride = stride
        self.pad = pad
        self.dilation = 2 if dilation else 1

        # if the bias is to be used, initialize it to 0
        if self.use_bias:
            self.bias = th.nn.Parameter(th.FloatTensor(c_out).fill_(0))

        # (product of kernel size) * number of input channels = (number of incoming neurons) ?
        fan_in = prod(_pair(k_size)) * c_in  # value of fan_in
        # scaling factor
        self.scale = sqrt(2) / sqrt(fan_in)

    def forward(self, x):
        """
        forward pass of the network
        :param x: input
        :return: y => output
        """
        from torch.nn.functional import conv2d

        return conv2d(input=x,
                      weight=self.weight * self.scale,  # scale the weight on runtime EACH TIME!
                      bias=self.bias if self.use_bias else None,
                      stride=self.stride,
                      padding=self.pad,
                      dilation=self.dilation)

    def extra_repr(self):
        return ", ".join(map(str, self.weight.shape))


class GatedConv2d(th.nn.Module):
    def __init__(self, c_in, c_out, k_size, stride=1, pad=0, bias=True, activation=None, dilation=False):
        super(GatedConv2d, self).__init__()

        dilation = 2 if dilation else 1

        self.conv_gating = Conv2d(c_in, c_out, k_size, stride=stride, padding=pad, bias=bias, dilation=dilation)
        self.conv_feature = Conv2d(c_in, c_out, k_size, stride=stride, padding=pad, bias=bias, dilation=dilation)
        assert activation is not None, "You must specify the activation!"
        self.activation = get_activation(activation)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        """
        forward pass of the layer
        :param x: input
        :return: y => output
        """
        gating = self.sigmoid(self.conv_gating(x))
        feature = self.conv_feature(x)
        if self.activation is not None:
            feature = self.activation(feature)
        y = feature * gating
        return y


class _equalized_gatedConv2d(th.nn.Module):
    """ 2D gated convolution with the concept of equalized learning rate
    EXPLANATION FROM PAPER:
    They explicitly scale the weights at runtime.
    They set the weight w_i = w_i/c where 'w_i' are weights and 'c' is the per-layer normalization constant
    from He's initializer.
    As a result, if some parameters have a larger dynamic range than others, they will take longer to adjust.
    On top of this, there is added the idea of  from "Free-Form Image Inpainting with Gated Convolution" (http://jiahuiyu.com/deepfill2/)
    """

    def __init__(self, c_in, c_out, k_size, stride=1, pad=0, bias=True, activation=None, dilation=False):
        """ conv2d with the concept of equalized learning rate
            Args:
                :param c_in: input channels
                :param c_out:  output channels
                :param k_size: kernel size (h, w) should be a tuple or a single integer
                :param stride: stride for conv
                :param pad: padding
                :param bias: whether to use bias or not
                :param activation: activation function used
        """
        """ constructor for the class """
        from torch.nn.modules.utils import _pair
        from numpy import sqrt, prod

        super(_equalized_gatedConv2d, self).__init__()

        self.conv_gating = _equalized_conv2d(c_in, c_out, k_size, stride=stride, pad=pad, bias=bias, dilation=dilation)
        self.conv_feature = _equalized_conv2d(c_in, c_out, k_size, stride=stride, pad=pad, bias=bias, dilation=dilation)
        assert activation is not None, "You must specify the activation!"
        self.activation = get_activation(activation)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        """
        forward pass of the network
        :param x: input
        :return: y => output
        """
        gating = self.sigmoid(self.conv_gating(x))
        feature = self.conv_feature(x)
        if self.activation is not None:
            feature = self.activation(feature)
        y = feature * gating
        return y


class _equalized_deconv2d(th.nn.Module):
    """ Transpose convolution using the equalized learning rate
        Args:
            :param c_in: input channels
            :param c_out: output channels
            :param k_size: kernel size
            :param stride: stride for convolution transpose
            :param pad: padding
            :param bias: whether to use bias or not
    """

    def __init__(self, c_in, c_out, k_size, stride=1, pad=0, bias=True):
        """ constructor for the class """
        from torch.nn.modules.utils import _pair
        from numpy import sqrt

        super(_equalized_deconv2d, self).__init__()

        # define the weight and bias if to be used
        self.weight = th.nn.Parameter(th.nn.init.normal_(
            th.empty(c_in, c_out, *_pair(k_size))
        ))

        self.use_bias = bias
        self.stride = stride
        self.pad = pad

        if self.use_bias:
            self.bias = th.nn.Parameter(th.FloatTensor(c_out).fill_(0))

        fan_in = c_in  # value of fan_in for deconv
        self.scale = sqrt(2) / sqrt(fan_in)

    def forward(self, x):
        """
        forward pass of the layer
        :param x: input
        :return: y => output
        """

        return conv_transpose2d(input=x,
                                weight=self.weight * self.scale,  # scale the weight on runtime EACH TIME!
                                bias=self.bias if self.use_bias else None,
                                stride=self.stride,
                                padding=self.pad)

    def extra_repr(self):
        return ", ".join(map(str, self.weight.shape))


class ResidualBlock(th.nn.Module):
    '''
    Residual block with gated convolutions.
    '''

    def __init__(self, dim, norm_layer, activation=ReLU(True), use_dropout=False, dilation=False, padding=1,
                 use_eql=True):
        super(ResidualBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, norm_layer, activation, use_dropout, dilation, padding, use_eql)

    def build_conv_block(self, dim, norm_layer, activation, use_dropout, dilation, padding, use_eql):
        conv_block = []
        p = padding if not dilation else 2

        if use_eql:
            conv = _equalized_gatedConv2d(dim, dim, 3, pad=p, activation=activation, dilation=dilation)
        else:
            conv = GatedConv2d(dim, dim, 3, pad=p, activation=activation, dilation=dilation)
        if norm_layer == AdaptiveInstanceNorm2d:
            conv_block += [conv, norm_layer(dim)]
        else:
            conv_block += [conv, norm_layer()]
        if use_dropout:
            conv_block += [Dropout(0.5)]

        return Sequential(*conv_block)

    def forward(self, x):
        conv_block_res = self.conv_block(x)
        out = x + conv_block_res
        return out


def get_activation(name):
    name = name.lower()
    if name == 'lrelu':
        return LeakyReLU(0.2)
    elif name == 'relu':
        return ReLU()
    elif name == 'none':
        return None
    else:
        raise Exception('Unspecified activation ({}). Implemented onlt Leaky ReLU and ReLU.'.format(name))


class BilinearInterpolation(th.nn.Module):
    def __init__(self, scale_factor, mode='bilinear'):
        super(BilinearInterpolation, self).__init__()
        self.interp = th.nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)
        return x


class UNetDown(th.nn.Module):
    def __init__(self, in_channels, out_channels, use_eql, activation='relu', downsampling='max',
                 normalization='batch_norm', ks=3, pad=1):
        super(UNetDown, self).__init__()

        if use_eql:
            self.conv_1 = _equalized_conv2d(in_channels, in_channels, (ks, ks), pad=pad, bias=True)
            self.conv_2 = _equalized_conv2d(in_channels, out_channels, (ks, ks), pad=pad, bias=True)
        else:
            self.conv_1 = Conv2d(in_channels, in_channels, (ks, ks), padding=pad, bias=True)
            self.conv_2 = Conv2d(in_channels, out_channels, (ks, ks), padding=pad, bias=True)

        # define activation
        activation = activation.lower()
        if activation == 'relu':
            self.activation = ReLU()
        elif activation in ['lrelu', 'leaky_relu', 'leakyrelu']:
            self.lrelu = LeakyReLU(0.2)
        else:
            raise NotImplementedError(
                'Activation {} is not implemented for UNet downsampling block.'.format(activation))

        # define downsampler
        donwsampling = downsampling.lower()
        if donwsampling == 'max':
            self.downsampler = MaxPool2d(2, 2)
        elif downsampling == 'avg':
            self.downsampler = AvgPool2d(2, 2)
        else:
            raise NotImplementedError('Downsampling "{}" is not implemented.'.format(downsampling))

        # define normalization
        normalization = normalization.lower() if normalization is not None else normalization
        self.use_norm = normalization is not None
        if normalization in ['bnorm', 'batch_norm', 'batchnorm']:
            self.normalization1 = BatchNorm2d(in_channels)
            self.normalization2 = BatchNorm2d(out_channels)
        elif normalization is None:
            self.normalization = None
        else:
            raise NotImplementedError('Normalization "{}" is not implemented.'.format(normalization))

    def forward(self, x):
        y = self.conv_1(x)
        if self.use_norm is not None:
            y = self.normalization1(y)
        y = self.activation(y)
        y = self.conv_2(y)
        if self.use_norm is not None:
            y = self.normalization2(y)
        features = y
        y = self.activation(features)
        y = self.downsampler(y)
        return y, features


class UNetUp(th.nn.Module):
    def __init__(self, in_channels, out_channels, use_eql, activation='relu', ks=3, pad=1):
        super(UNetUp, self).__init__()

        # number of filters is multiplied by 2 since it corresponds to the number of filters
        # in the downsampling stream and we will add those features to the upsampled ones
        # If confused, read about UNet architecture.
        if use_eql:
            self.deconv_1 = _equalized_deconv2d(2 * in_channels, in_channels, (2, 2), stride=2, bias=True)
            self.conv_2 = _equalized_conv2d(2 * in_channels, in_channels, (ks, ks), pad=pad, bias=True)
        else:
            self.deconv_1 = ConvTranspose2d(2 * in_channels, in_channels, (2, 2), stride=2, bias=True)
            self.conv_2 = Conv2d(2 * in_channels, in_channels, (ks, ks), padding=pad, bias=True)

        # define activation
        activation = activation.lower()
        if activation == 'relu':
            self.activation = ReLU()
        elif activation in ['lrelu', 'leaky_relu', 'leakyrelu']:
            self.lrelu = LeakyReLU(0.2)
        else:
            raise NotImplementedError(
                'Activation {} is not implemented for UNet downsampling block.'.format(activation))

    def forward(self, x, residual_connection):
        y = self.activation(self.deconv_1(x))
        y = th.cat((y, residual_connection), dim=1)
        y = self.conv_2(y)
        return y


class UNetCore(th.nn.Module):
    def __init__(self, in_channels, out_channels, use_eql, activation='relu'):
        super(UNetCore, self).__init__()

        conv_model = _equalized_conv2d if use_eql else Conv2d

        # define activation
        activation = activation.lower()
        if activation == 'relu':
            self.activation = ReLU
        elif activation in ['lrelu', 'leaky_relu', 'leakyrelu']:
            self.lrelu = LeakyReLU(0.2)
        else:
            raise NotImplementedError(
                'Activation {} is not implemented for UNet downsampling block.'.format(activation))

        # define model
        self.model = Sequential(
            conv_model(in_channels, out_channels, (3, 3), pad=1, bias=True),
            self.activation(),
            conv_model(out_channels, out_channels, (3, 3), pad=1, bias=True),
            self.activation(),
        )

    def forward(self, x):
        y = self.model(x)
        return y


class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='none', activation='relu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        # initialize fully connected layer
        if norm == 'sn':
            self.fc = SpectralNorm(nn.Linear(input_dim, output_dim, bias=use_bias))
        else:
            self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out
