import os
import random
import sys

from PIL import Image

tmp = '/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-1])
sys.path.append(tmp)

import re
import timeit

import time

import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.utils.spectral_norm as spectral_norm
from tensorboardX import SummaryWriter
from torch.nn import init, AvgPool2d, DataParallel

# This code is mainly taken from the original paper https://nvlabs.github.io/SPADE/, https://github.com/NVlabs/SPADE.
from torch.optim import Adam
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid, save_image

from models.DataTools import get_data_loader
from models.MaskEstimator import MaskEstimator
from models.Losses import WGAN_GP, KL_loss, L2MaskedLoss, VGGLoss, L1MaskedLoss
from models.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from models.networks import VAE_enc_only


def remove_module(weights):
    wo_module = OrderedDict()
    for k, v in weights.items():
        name = k.replace('module.', '')
        wo_module[name] = v
    return wo_module


# Returns a function that creates a normalization function
# that does not condition on semantic map
def get_nonspade_norm_layer(opt, norm_type='instance'):
    # helper function to get # output channels of the previous layer
    def get_out_channel(layer):
        if hasattr(layer, 'out_channels'):
            return getattr(layer, 'out_channels')
        return layer.weight.size(0)

    # this function will be returned
    def add_norm_layer(layer):
        nonlocal norm_type
        if norm_type.startswith('spectral'):
            print('Use spectral norm!!!')
            layer = spectral_norm(layer)
            norm_type = norm_type[len('spectral'):]

        # if subnorm_type == 'none' or len(subnorm_type) == 0:
        #    return layer

        # remove bias in the previous layer, which is meaningless
        # since it has no effect after normalization
        if getattr(layer, 'bias', None) is not None:
            delattr(layer, 'bias')
            layer.register_parameter('bias', None)

        if norm_type == 'batch':
            norm_layer = nn.BatchNorm2d(get_out_channel(layer), affine=True)
        elif norm_type == 'sync_batch':
            norm_layer = SynchronizedBatchNorm2d(get_out_channel(layer), affine=True)
        elif norm_type == 'instance':
            norm_layer = nn.InstanceNorm2d(get_out_channel(layer), affine=False)
        else:
            raise ValueError('normalization layer %s is not recognized' % norm_type)

        return nn.Sequential(layer, norm_layer)

    return add_norm_layer


class KeypointsDownsampler(nn.Module):
    def __init__(self, tgt_size, mode='bilinear'):
        super(KeypointsDownsampler, self).__init__()
        self.tgt_size = tgt_size
        self.mode = mode

    def forward(self, x):
        dwn = F.interpolate(x, self.tgt_size, mode=self.mode)
        return dwn


# Creates SPADE normalization layer based on the given configuration
# SPADE consists of two steps. First, it normalizes the activations using
# your favorite normalization method, such as Batch Norm or Instance Norm.
# Second, it applies scale and bias to the normalized output, conditioned on
# the segmentation map.
# The format of |config_text| is spade(norm)(ks), where
# (norm) specifies the type of parameter-free normalization.
#       (e.g. syncbatch, batch, instance)
# (ks) specifies the size of kernel in the SPADE module (e.g. 3x3)
# Example |config_text| will be spadesyncbatch3x3, or spadeinstance5x5.
# Also, the other arguments are
# |norm_nc|: the #channels of the normalized activations, hence the output dim of SPADE
# |label_nc|: the #channels of the input semantic map, hence the input dim of SPADE
class SPADE(nn.Module):
    def __init__(self, config_text, norm_nc, label_nc, opts):
        super().__init__()

        assert config_text.startswith('spade')
        parsed = re.search('spade(\D+)(\d)x\d', config_text)
        # param_free_norm_type = str(parsed.group(1))
        ks = 3

        if opts.param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif opts.param_free_norm_type == 'syncbatch':
            self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
        elif opts.param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % opts.param_free_norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, segmap):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out


# ResNet block that uses SPADE.
# It differs from the ResNet block of pix2pixHD in that
# it takes in the segmentation map as input, learns the skip connection if necessary,
# and applies normalization first and then convolution.
# This architecture seemed like a standard architecture for unconditional or
# class-conditional GAN architecture using residual block.
# The code was originally inspired from https://github.com/LMescheder/GAN_stability.
# It is taken from https://github.com/NVlabs/SPADE/blob/master/models/networks/architecture.py
class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, args):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)
        n_cond = self.get_n_cond_channels(args)

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified
        if 'spectral' in args.norm_G:
            print('SPECTRAL norm in G!')
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        spade_config_str = args.norm_G.replace('spectral', '')
        self.norm_0 = SPADE(spade_config_str, fin, n_cond, args)
        self.norm_1 = SPADE(spade_config_str, fmiddle, n_cond, args)
        if self.learned_shortcut:
            self.norm_s = SPADE(spade_config_str, fin, n_cond, args)

    def get_n_cond_channels(self, args):
        n = 0
        if args.use_bg:
            n += 3
        if args.use_kpts:
            n += 17
        if args.use_shape:
            n += 1
        if args.enc2spade:
            n += 1
        return n

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg):
        x_s = self.shortcut(x, seg)

        dx = self.conv_0(self.actvn(self.norm_0(x, seg)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg)))

        out = x_s + dx

        return out

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def print_network(self):
        if isinstance(self, list):
            self = self[0]
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print('Network [%s] was created. Total number of parameters: %.1f million. '
              'To see the architecture, do print(network).'
              % (type(self).__name__, num_params / 1000000))

    def init_weights(self, init_type='normal', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)


class DummyGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
        parser.add_argument('--num_upsampling_layers',
                            choices=('normal', 'more', 'most'), default='normal',
                            help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator")

        return parser

    def __init__(self, args):
        super().__init__()
        self.args = args
        nf = args.ngf
        self.max_mult = 2 ** (args.depth - 1)

        self.sw, self.sh = self.compute_latent_vector_size(args)

        if args.use_vae:
            # In case of VAE, we will sample from random z vector
            self.fc = nn.Linear(args.z_dim, self.max_mult * nf * self.sw * self.sh)
        else:
            # Otherwise, we make the network deterministic by starting with
            # downsampled segmentation map instead of random z
            self.fc = nn.Conv2d(self.args.semantic_nc, self.max_mult * nf, 3, padding=1)

        head_0 = SPADEResnetBlock(self.max_mult * nf, self.max_mult * nf, args)  # +up=8x8
        head_final_layer = nn.Conv2d(self.max_mult * nf, 3, 3, padding=1)

        # self.G_middle_0 = SPADEResnetBlock(max_mult * nf, max_mult * nf, args) # +up=16x16
        # self.G_middle_1 = SPADEResnetBlock(max_mult * nf, max_mult * nf, args) # +up=32x32

        self.layers_up = nn.ModuleList([head_0])
        for i in range(args.depth - 1):
            cin = int(self.max_mult / (2 ** i)) * nf
            cout = int(self.max_mult / (2 ** (i + 1))) * nf
            self.layers_up.append(SPADEResnetBlock(cin, cout, args))

        final_nc = nf

        self.final_layers = nn.ModuleList([head_final_layer])
        for i in range(args.depth - 1):
            cin = int(self.max_mult / (2 ** (i + 1))) * nf
            self.final_layers.append(nn.Conv2d(cin, 3, 3, padding=1))

        self.up = lambda x: nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

    def compute_latent_vector_size(self, opt):
        num_up_layers = opt.depth  # should be 5
        sw = opt.image_size // (2 ** num_up_layers)
        sh = sw

        return sw, sh  # should be 4x4

    def __progressive_downsampling(self, real_batch, depth, alpha=1.0):
        """
        private helper for downsampling the original images in order to facilitate the
        progressive growing of the layers.
        :param real_batch: batch of real samples
        :param depth: depth at which training is going on
        :param alpha: current value of the fader alpha
        :return: real_samples => modified real batch of samples
        """

        from torch.nn import AvgPool2d
        from torch.nn.functional import interpolate

        # downsample the real_batch for the given depth
        down_sample_factor = int(np.power(2, self.args.depth - depth - 1))
        prior_downsample_factor = max(int(np.power(2, self.args.depth - depth)), 0)

        ds_real_samples = AvgPool2d(down_sample_factor)(real_batch)

        if depth > 0:
            prior_ds_real_samples = interpolate(AvgPool2d(prior_downsample_factor)(real_batch),
                                                scale_factor=2)
        else:
            prior_ds_real_samples = ds_real_samples

        # real samples are a combination of ds_real_samples and prior_ds_real_samples
        real_samples = (alpha * ds_real_samples) + ((1 - alpha) * prior_ds_real_samples)

        # return the so computed real_samples
        return real_samples

    def resize_input(self, inp, downsampling_factor):
        resized = self.__progressive_downsampling(inp, downsampling_factor)
        return resized

    def forward(self, input, z=None, depth=None, alpha=None):

        cond_input = []
        for i in range(depth + 1):
            downsampled = self.resize_input(input, i)
            cond_input.append(downsampled)

        if self.args.use_vae:
            # we sample z from unit normal and reshape the tensor
            if z is None:
                z = torch.randn(input.size(0), self.args.z_dim,
                                dtype=torch.float32, device=input.device)
            x = self.fc(z)
            x = x.view(-1, self.max_mult * self.args.ngf, self.sh, self.sw)
        else:
            # we downsample input and run convolution
            x = F.interpolate(input, size=(self.sh, self.sw))
            x = self.fc(x)

        # 4x4

        if depth == 0:  # 4x4
            x = self.up(x)  # 8x8
            x = self.layers_up[0](x, cond_input[0])
            x = self.final_layers[0](F.leaky_relu(x, 2e-1))
            # x = torch.tanh(x)  # 8x8
            x = torch.sigmoid(x)

        else:  # 4x4
            for i, up_layer in enumerate(self.layers_up[:depth]):
                x = self.up(x)
                x = up_layer(x, cond_input[i])

            # upsamle the output such that 'upsampled' has the target resolution
            upsampled = self.up(x)

            # upsampled output of the previous layer
            residual = self.final_layers[depth - 1](upsampled)
            # residual = torch.tanh(residual)
            residual = torch.sigmoid(residual)

            # output of the most recent layer
            straight = self.layers_up[depth](upsampled, cond_input[depth])
            straight = self.final_layers[depth](straight)
            # straight = torch.tanh(straight)
            straight = torch.sigmoid(straight)

            # linear combination for smooth blending
            x = (alpha * straight) + ((1 - alpha) * residual)

        return x


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--n_layers_D', type=int, default=3,
                            help='# layers in each discriminator')
        return parser

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.use_rock, self.use_pred = False, False

        kw = 4
        padw = 1  # int(np.ceil((kw - 1.0) / 2))
        nf = args.ndf
        input_nc = self.compute_D_input_nc(args)

        norm_layer = get_nonspade_norm_layer(args, args.norm_D)
        self.layers = nn.ModuleList([nn.Sequential(nn.Conv2d(input_nc, nf, kernel_size=kw, stride=2, padding=padw),
                                                   nn.LeakyReLU(0.2, False))])
        to_features = [None]

        for n in range(1, args.depth):
            nf_prev = nf
            nf = min(nf * 2, 512)
            self.layers.append(nn.Sequential(norm_layer(nn.Conv2d(nf_prev, nf, kernel_size=kw,
                                                                  stride=2, padding=padw)),
                                             nn.LeakyReLU(0.2, False)))
            to_features += [nn.Conv2d(input_nc, nf_prev, 1, 1)]

        self.layers.append(nn.Sequential(nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)))
        to_features += [None]

        self.to_features = nn.ModuleList(to_features)
        self.temporary_downsampler = AvgPool2d(2)

    def compute_D_input_nc(self, args):
        input_nc = 3  # for RGB
        if args.use_kpts and not args.dis_use_no_kpts:
            input_nc += 17
        if args.use_shape:
            input_nc += 1
        if args.enc2spade:
            input_nc += 1
        return input_nc

    def forward_orig(self, input):
        results = [input]
        for submodel in self.children():
            intermediate_output = submodel(results[-1])
            results.append(intermediate_output)

        get_intermediate_features = not self.args.no_ganFeat_loss
        if get_intermediate_features:
            return results[1:]
        else:
            return results[-1]

    def forward(self, x, depth, alpha, get_features=False):
        first_layer_id = -(depth + 2)
        features = []

        if depth == 0:
            to_fm = self.to_features[first_layer_id]
            if to_fm is None:
                y = x
            else:
                y = to_fm(x)
            for layer in self.layers[first_layer_id:]:  # always take the last layer
                y = layer(y)
                features.append(y)

        else:
            to_fm = self.to_features[first_layer_id]
            residual = self.to_features[first_layer_id + 1](self.temporary_downsampler(x))
            if to_fm is not None:
                x = to_fm(x)
                straight = self.layers[first_layer_id](x)
            else:
                straight = self.layers[first_layer_id](x)

            y = (alpha * straight) + ((1 - alpha) * residual)
            features.append(y)

            for layer in self.layers[first_layer_id + 1:]:  # always take the last layer
                y = layer(y)
                features.append(y)

        if get_features:
            return features
        return y


class Dummy_GAN:
    def __init__(self, args, debug=False):

        if args.single_gpu:
            args.multi_gpu = False
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        self.debug = debug
        if self.debug:
            print('DEBUG MODE')
        self.args = args
        self.device = args.device
        self.args.log_dir = '{}{}/'.format(args.log_dir, args.name)
        self.args.sample_dir = '{}{}/'.format(args.sample_dir, args.name)
        self.args.save_dir = '{}{}/'.format(args.save_dir, args.name)
        self.FloatTensor = torch.cuda.FloatTensor if self.device == 'cuda' else torch.FloatTensor

        args.input_nc, args.input_blending = self.get_input_nc(args)

        args.semantic_nc = self.get_n_cond_channels(args)
        self.args.semantic_nc = args.semantic_nc

        # create tensorboardX summary writer
        self.writer = SummaryWriter('./summary/{}/'.format(args.name))

        # keypoints downsampler
        self.me_input_resizer = KeypointsDownsampler(tgt_size=64, mode='bilinear')

        # set networks
        self.generator = DummyGenerator(args).to(self.device)
        self.discriminator_scene = NLayerDiscriminator(args).to(self.device)
        self.appearance_encoder = VAE_enc_only(in_channels=3, code_dim_base=args.z_dim).to(self.device)

        # Optionally, use pretrained encoder and freeze it.
        if self.args.use_pretrained_encoder:
            encoder_w = torch.load(self.args.encoder_weights, map_location=lambda storage, loc: storage)
            encoder_w = remove_module(encoder_w)
            self.appearance_encoder.load_state_dict(encoder_w)
            for param in self.appearance_encoder.parameters():
                param.requires_grad = False

        # Optionally, use pretrained mask estimator and freeze it.
        if self.args.use_pretrained_mask_estimator:
            chkpt = torch.load(self.args.mask_estimator_chkpt, map_location=lambda storage, loc: storage)
            me_args = chkpt['args']
            self.mask_estimator = MaskEstimator(me_args, 17).to(self.device)
            me_weights = remove_module(chkpt['model'])
            self.mask_estimator.load_state_dict(me_weights)
            for param in self.mask_estimator.parameters():
                param.requires_grad = False
        else:
            self.mask_estimator = MaskEstimator(args, 17).to(self.device)

        '''
        if self.args.multi_gpu and torch.cuda.device_count() > 1:
            self.generator = DataParallel(self.generator)
            self.appearance_encoder = DataParallel(self.appearance_encoder)
            self.discriminator_scene = DataParallel(self.discriminator_scene)
            self.mask_estimator = DataParallel(self.mask_estimator)
        '''
        if self.device != 'cpu' and torch.cuda.device_count() > 1 and torch.distributed.is_available():
            torch.distributed.init_process_group(backend='nccl',  # 'distributed backend'
                                                 init_method='tcp://127.0.0.1:9999',  # distributed training init method
                                                 world_size=1,  # number of nodes for distributed training
                                                 rank=0)  # distributed training node rank
            self.generator = torch.nn.parallel.DistributedDataParallel(self.generator, find_unused_parameters=True)
            self.appearance_encoder = DataParallel(self.appearance_encoder)
            self.discriminator_scene = torch.nn.parallel.DistributedDataParallel(self.discriminator_scene,
                                                                                 find_unused_parameters=True)
            self.mask_estimator = DataParallel(self.mask_estimator)

        # if code is to be run on GPU, we can use DataParallel:
        self.n_gpus = torch.cuda.device_count()
        if self.device == 'cuda':
            print('CUDA available. Number of GPUs: {}.'.format(self.n_gpus))
        else:
            print('No GPU available!!!')

        # get parameters
        param_G = list(self.generator.parameters())
        if not self.args.use_pretrained_encoder:
            param_G += list(self.appearance_encoder.parameters())
        if not self.args.use_pretrained_mask_estimator:
            param_G += list(self.mask_estimator.parameters())

        # set optimizers
        self.optim_G = Adam(param_G, lr=args.lr_gen, betas=(args.beta_1, args.beta_2),
                            eps=args.eps)  # has encoder and blending net parameters
        self.optim_D = Adam(self.discriminator_scene.parameters(), lr=args.lr_dis,
                            betas=(args.beta_1, args.beta_2), eps=args.eps)

        if self.args.load_weights or self.args.resume:
            # models
            gen_weights = remove_module(torch.load(args.spade_g_weights, map_location=self.device))
            self.generator.load_state_dict(gen_weights)
            print('Loaded generator.')
            dis_weights = remove_module(torch.load(args.spade_d_weights, map_location=self.device))
            self.discriminator_scene.load_state_dict(dis_weights)
            print('Loaded discriminator.')
            if not self.args.use_pretrained_mask_estimator:
                me_weights = remove_module(torch.load(args.spade_blending_weights, map_location=self.device))
                self.mask_estimator.load_state_dict(me_weights)
                print('Loaded mask estimator.')
            if not self.args.use_pretrained_encoder:
                encoder_w = remove_module(torch.load(args.encoder_weights, map_location=self.device))
                self.appearance_encoder.load_state_dict(encoder_w)
                print('Loaded encoder.')
            # optimizers
            gen_optim_weights = remove_module(torch.load(args.spade_g_optim_weights, map_location=self.device))
            self.optim_G.load_state_dict(gen_optim_weights)
            print('Loaded generator optimizer.')
            dis_optim_weights = torch.load(args.spade_d_optim_weights, map_location=self.device)
            self.optim_D.load_state_dict(dis_optim_weights)
            print('Loaded discriminator optimizer.')

        # LOSSES
        # GAN loss
        self.loss_gan = WGAN_GP(self.discriminator_scene, use_gp=True)
        # KLD loss for latent vector
        self.loss_kl = KL_loss()
        # reconstruction loss on G
        self.loss_G = {} if args.loss_G_spade == 'none' else {ln: self.get_loss(ln) for ln in args.loss_G_spade}
        # reconstruction loss on the mask
        self.loss_mask = {ln: self.get_loss(ln) for ln in args.mask_loss}
        # FM losses
        if args.use_fm_loss:
            self.fm_loss = torch.nn.L1Loss()
        if args.use_vgg_loss:
            self.vgg_loss = VGGLoss(args.device)
        # encoder cycle loss
        if args.encoder_cycle_loss:
            self.encoder_cycle_loss = L1MaskedLoss(average=True, foreground=True)

        self.smallest_size = int(self.args.image_size / np.power(2, self.args.depth - 1))
        print('Smallest size: {}'.format(self.smallest_size))

    def get_loss(self, loss_name):
        loss_name = loss_name.lower()
        if loss_name == 'l1_masked':
            return L1MaskedLoss(average=True, foreground=True)
        elif loss_name == 'l2_masked':
            return L2MaskedLoss(average=True, foreground=True)
        elif loss_name == 'l1':
            return nn.L1Loss()
        elif loss_name == 'l2':
            return nn.MSELoss()
        else:
            raise Exception('Unsupported loss! Passed loss dataset_name: {}.'.format(loss_name))

    def get_input_nc(self, args):
        input_blending = 0
        self.cond_input = []
        if args.use_bg:  # use masked background in SPADE
            self.cond_input.append('image_masked')
        if args.use_shape:
            input_blending += 1
            self.cond_input.append('mask_orig')
        if args.use_kpts:
            input_blending += 17
            self.cond_input.append('mask_keypoint')
        if args.enc2spade:
            self.cond_input.append('encoder')

        # input to spade consists of RGB + conditional input
        input_spade = input_blending + 3 + 1 if args.enc2spade else 0

        return input_spade, input_blending

    def get_n_cond_channels(self, args):
        n = 0
        if args.use_bg:
            n += 3
        if args.use_kpts:
            n += 17
        if args.use_shape:
            n += 1
        if args.enc2spade:
            n += 1
        return n

    def __progressive_downsampling(self, real_batch, depth, alpha=1.0):
        """
        private helper for downsampling the original images in order to facilitate the
        progressive growing of the layers.
        :param real_batch: batch of real samples
        :param depth: depth at which training is going on
        :param alpha: current value of the fader alpha
        :return: real_samples => modified real batch of samples
        """

        from torch.nn import AvgPool2d
        from torch.nn.functional import interpolate

        # downsample the real_batch for the given depth
        down_sample_factor = int(np.power(2, self.args.depth - depth - 1))
        prior_downsample_factor = max(int(np.power(2, self.args.depth - depth)), 0)

        ds_real_samples = AvgPool2d(down_sample_factor)(real_batch)

        if depth > 0:
            prior_ds_real_samples = interpolate(AvgPool2d(prior_downsample_factor)(real_batch),
                                                scale_factor=2)
        else:
            prior_ds_real_samples = ds_real_samples

        # real samples are a combination of ds_real_samples and prior_ds_real_samples
        real_samples = (alpha * ds_real_samples) + ((1 - alpha) * prior_ds_real_samples)

        # return the so computed real_samples
        return real_samples

    def resize_input(self, inp, downsampling_factor):
        # data
        resized = self.__progressive_downsampling(inp, downsampling_factor)

        return resized

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std) + mu

    def encode_z(self, real_person, only_code=False):
        if self.args.use_pretrained_encoder:
            with torch.no_grad():
                real_person_resized = F.interpolate(real_person, self.args.encoder_input_size, mode='bilinear')
            z = self.appearance_encoder(real_person_resized.to(self.device), return_mu_logvar=False)
            del real_person_resized
            return z
        else:
            mu, logvar = self.appearance_encoder(real_person)
            z = self.reparameterize(mu, logvar)
            if only_code:
                return z
            else:
                return z, mu, logvar

    def sample_batch(self, batch, n_samples):
        bs = batch['image'].shape[0]
        start_id = random.randint(0, max([0, bs - n_samples]))
        sampled_batch = {}
        for key, val in batch.items():
            sampled_batch[key] = val[start_id:(start_id + n_samples)]
        return sampled_batch

    def train_gen(self, keypoints, target_scene, mask, depth, alpha):
        z = self.encode_z(self.target_person_full) if self.args.use_vae else None
        if isinstance(z, tuple):  # and not self.args.use_pretrained_encoder:
            z, mu, logvar = z
            kld = self.loss_kl(mu, logvar) * self.args.kld_lbd

        # get generated image
        with torch.no_grad():
            estimated_mask = self.estimate_mask(self.mask_estimator, keypoints)
            estimated_mask = F.interpolate(estimated_mask, target_scene.shape[-2:], mode='bilinear')
        masked_images = target_scene * (1 - estimated_mask)
        cond_input = torch.cat((masked_images, keypoints), dim=1)
        gen_output = self.generator(cond_input, z, depth, alpha)
        gen_scene = gen_output * estimated_mask + target_scene * (1 - estimated_mask)

        if not self.args.gan_loss_person_only:
            # OPTION 1: Compute GAN loss on the complete image
            if not self.args.dis_use_no_kpts:
                target_scene_concat = torch.cat((target_scene, keypoints), dim=1)
                gen_scene_concat = torch.cat((gen_scene, keypoints), dim=1)
            else:
                target_scene_concat = target_scene
                gen_scene_concat = gen_scene
        else:
            # OPTION 2: Compute GAN loss on the generated part only
            gen_part = gen_output * estimated_mask
            target_part = target_scene * estimated_mask
            if self.args.dis_use_no_kpts:
                target_scene_concat = target_part
                gen_scene_concat = gen_part
            else:
                target_scene_concat = torch.cat((target_part, keypoints), dim=1)
                gen_scene_concat = torch.cat((gen_part, keypoints), dim=1)

        loss = self.loss_gan.gen_loss(target_scene_concat, gen_scene_concat, depth, alpha) * self.args.gan_lbd
        self.latest_losses['gen'][self.loss_gan.name] = loss.item()

        # luminance loss
        if self.args.use_luminance_loss:
            fg = gen_output * estimated_mask
            bg = target_scene * (1 - estimated_mask)
            luminance_loss = self.luminary_loss(fg, bg) * self.args.luminance_loss_lbd
            loss += luminance_loss
            self.latest_losses['gen']['luminance'] = luminance_loss.item()

        # if z is not None, add KLD loss
        if z is not None and not self.args.use_pretrained_encoder:
            loss += kld
            self.latest_losses['gen']['kld'] = kld.item()

        # reconstruction loss
        for name, loss_fnc in self.loss_G.items():
            gen_person_from_scene = gen_output * estimated_mask
            target_person = target_scene * estimated_mask
            tmp_loss = loss_fnc(gen_person_from_scene, target_person, mask=self.target_mask)
            if name != 'edge':
                tmp_loss = tmp_loss * self.args.rec_loss_lbd
            loss = loss + tmp_loss
            self.latest_losses['gen'][name] = tmp_loss.item()

        # feature matching loss
        if self.args.use_fm_loss:
            GAN_Feat_loss = self.FloatTensor(1).fill_(0)
            pred_fake = self.discriminator_scene(gen_scene_concat, depth, alpha, get_features=True)
            pred_real = self.discriminator_scene(target_scene_concat, depth, alpha, get_features=True)
            for j in range(len(pred_fake) - 1):  # for each layer output
                unweighted_loss = self.fm_loss(pred_fake[j], pred_real[j].detach())
                GAN_Feat_loss += unweighted_loss * self.args.fm_lbd
            self.latest_losses['gen']['GAN_Feat'] = GAN_Feat_loss.item()
            loss = loss + GAN_Feat_loss

        # VGG loss
        if self.args.use_vgg_loss:
            vgg_inp_gen = gen_output * estimated_mask
            vgg_inp_real = target_scene * estimated_mask
            vgg_loss = self.vgg_loss(fake=vgg_inp_gen, real=vgg_inp_real) * self.args.vgg_lbd
            self.latest_losses['gen']['VGG'] = vgg_loss.item()
            loss = loss + vgg_loss

        # encoder cycle loss
        if self.args.encoder_cycle_loss:
            '''
            Matches codes of real samples and generated samples obtained by encoder
            Currently requires use of pretrained encoder
            '''
            assert self.args.use_pretrained_encoder

            real_full_masked = self.target_person_full * self.target_mask_full
            real_input = F.interpolate(real_full_masked, self.args.encoder_input_size, mode='bilinear')

            mask_resized = F.interpolate(self.target_mask_full, self.args.encoder_input_size, mode='bilinear')
            gen_input = F.interpolate(gen_scene, self.args.encoder_input_size, mode='bilinear') * mask_resized

            size_weight = 1.0  # max([0.3, gen_person.shape[-1] / self.target_person_full.shape[-1]])

            z_real = self.encode_z(real_input, only_code=True)
            z_gen = self.encode_z(gen_input, only_code=True)
            e_loss = F.l1_loss(z_gen, z_real)
            e_loss = e_loss * self.args.encoder_cycle_loss_lbd * size_weight
            self.latest_losses['gen']['enc_cycle'] = e_loss.item()
            loss = loss + e_loss

        self.optim_G.zero_grad()
        loss.backward()
        self.optim_G.step()

        del loss
        del masked_images
        del estimated_mask
        del gen_output
        del gen_scene
        del gen_scene_concat

    def train_dis(self, keypoints, target_scene, depth, alpha):
        z = self.encode_z(self.target_person_full) if self.args.use_vae else None
        if z is not None:
            if not isinstance(z, torch.Tensor):
                z, _, _ = z
            z = z.detach()

        # get generated image
        with torch.no_grad():
            estimated_mask = self.estimate_mask(self.mask_estimator, keypoints)
            estimated_mask = F.interpolate(estimated_mask, target_scene.shape[-2:], mode='bilinear')
        masked_images = target_scene * (1 - estimated_mask)
        cond_input = torch.cat((masked_images, keypoints), dim=1)
        gen_output = self.generator(cond_input, z, depth, alpha).detach()

        gen_scene = gen_output * estimated_mask + target_scene * (1 - estimated_mask)

        self.latest_person = gen_output
        self.latest_scene = gen_scene
        self.latest_mask = estimated_mask
        self.latest_masked_input = masked_images

        # detach scene for computation of the scene discriminator
        gen_scene = gen_scene.detach()

        if not self.args.gan_loss_person_only:
            # OPTION 1: Compute GAN loss on the complete image
            if not self.args.dis_use_no_kpts:
                target_scene_concat = torch.cat((target_scene, keypoints), dim=1)
                gen_scene_concat = torch.cat((gen_scene, keypoints), dim=1)
            else:
                target_scene_concat = target_scene
                gen_scene_concat = gen_scene
        else:
            # OPTION 2: Compute GAN loss on the generated part only
            gen_part = gen_output * estimated_mask
            target_part = target_scene * estimated_mask
            if self.args.dis_use_no_kpts:
                target_scene_concat = target_part
                gen_scene_concat = gen_part
            else:
                target_scene_concat = torch.cat((target_part, keypoints), dim=1)
                gen_scene_concat = torch.cat((gen_part, keypoints), dim=1)

        loss_scene = self.loss_gan.dis_loss(target_scene_concat, gen_scene_concat, depth, alpha) * self.args.gan_lbd
        self.latest_losses['dis']['wgan-gp'] = loss_scene.item()

        self.optim_D.zero_grad()
        loss_scene.backward()
        self.optim_D.step()

        del loss_scene
        del masked_images
        del estimated_mask
        del gen_output
        del gen_scene
        del gen_scene_concat

    def train_one_step(self, keypoints, target_scene, mask, depth, alpha):
        self.latest_losses = {'gen': {}, 'dis': {}}
        self.train_gen(keypoints, target_scene, mask, depth, alpha)
        self.train_dis(keypoints, target_scene, depth, alpha)

    def save_results(self, i, current_depth, epoch, alpha):
        # TRAINING
        gen_person_file = os.path.join(self.args.sample_dir, "gen_" + str(current_depth) +
                                       "_" + str(epoch) + "_" +
                                       str(i) + ".jpg")
        shape_file = os.path.join(self.args.sample_dir, "shape_" + str(current_depth) +
                                  "_" + str(epoch) + "_" +
                                  str(i) + ".jpg")
        masked_input_file = os.path.join(self.args.sample_dir, "masked_input_" + str(current_depth) +
                                         "_" + str(epoch) + "_" +
                                         str(i) + ".jpg")
        scene_file = os.path.join(self.args.sample_dir, "scene_" + str(current_depth) +
                                  "_" + str(epoch) + "_" +
                                  str(i) + ".jpg")

        n_row = int(int(np.sqrt(self.latest_person.shape[0]) / 2)) * 2  # how many samples per row
        n_row = max([n_row, 2])

        self.latest_person = self.latest_person[:int(n_row ** 2)]
        self.latest_mask = self.latest_mask[:int(n_row ** 2)]
        self.latest_scene = self.latest_scene[:int(n_row ** 2)]
        self.latest_masked_input = self.latest_masked_input[:int(n_row ** 2)]

        # comparison of person
        person_comp = torch.empty_like(self.latest_person)
        person_comp[1::2] = self.latest_person[:int((n_row ** 2) / 2)]
        person_comp[0::2] = self.target_person[:int((n_row ** 2) / 2)]

        # comparison of shape
        shape_comp = torch.empty_like(self.latest_mask)
        shape_comp[1::2] = self.latest_mask[:int((n_row ** 2) / 2)]
        shape_comp[0::2] = self.target_mask[:int((n_row ** 2) / 2)]

        # comparison of scene
        scene_comp = torch.empty_like(self.latest_person)
        scene_comp[1::2] = self.latest_scene[:int((n_row ** 2) / 2)]
        scene_comp[0::2] = self.target_scene[:int((n_row ** 2) / 2)]

        # comparison of masked inputs
        input_comp = torch.empty_like(self.latest_masked_input)
        input_comp[1::2] = self.latest_masked_input[:int((n_row ** 2) / 2)]
        input_comp[0::2] = self.target_scene[:int((n_row ** 2) / 2)]

        grid_person = make_grid(person_comp, nrow=n_row)
        save_image(grid_person, gen_person_file)
        grid_shape = make_grid(shape_comp, nrow=n_row)
        save_image(grid_shape, shape_file)
        grid_scene = make_grid(scene_comp, nrow=n_row)
        save_image(grid_scene, scene_file)
        grid_input = make_grid(input_comp, nrow=n_row)
        save_image(grid_input, masked_input_file)

        # VALIDATION
        self.generator.eval()
        if self.use_val:
            with torch.no_grad():
                z = self.encode_z(self.val_person_full) if self.args.use_vae else None
                if z is not None:
                    if not isinstance(z, torch.Tensor):
                        z, _, _ = z
                    z = z.detach()

                # get generated image
                estimated_mask = self.estimate_mask(self.mask_estimator, self.val_keypoints)
                estimated_mask = F.interpolate(estimated_mask, self.val_scene.shape[-2:], mode='bilinear').cpu()
                mask_val = estimated_mask

            masked_images = self.val_scene * (1 - estimated_mask)
            cond_input = torch.cat((masked_images, self.val_keypoints), dim=1).to(self.device)
            gen_output = self.generator(cond_input, z, current_depth, alpha).detach().cpu()
            gen_person_val = gen_output

            del cond_input
            del masked_images

            gen_scene_val = gen_output * estimated_mask + self.val_scene * (1 - estimated_mask)
            del gen_output
            del estimated_mask

            gen_person_file_val = os.path.join(self.args.sample_dir, "gen_val_" + str(current_depth) +
                                               "_" + str(epoch) + "_" + str(i) + ".jpg")
            shape_file_val = os.path.join(self.args.sample_dir, "shape_val_" + str(current_depth) +
                                          "_" + str(epoch) + "_" +
                                          str(i) + ".jpg")
            scene_file_val = os.path.join(self.args.sample_dir, "scene_val_" + str(current_depth) +
                                          "_" + str(epoch) + "_" +
                                          str(i) + ".jpg")

            n_row = int(np.ceil(np.sqrt(self.latest_person.shape[0])) / 2) * 2  # how many samples per row

            self.latest_person = self.latest_person[:int(n_row ** 2)]
            self.latest_mask = self.latest_mask[:int(n_row ** 2)]
            self.latest_scene = self.latest_scene[:int(n_row ** 2)]

            # comparison of person
            person_comp = torch.empty_like(self.latest_person)
            person_comp[1::2] = gen_person_val[:int((n_row ** 2) / 2)]
            person_comp[0::2] = self.val_person[:int((n_row ** 2) / 2)]
            del gen_person_val

            # comparison of shape
            shape_comp = torch.empty_like(self.latest_mask)
            shape_comp[1::2] = mask_val[:int((n_row ** 2) / 2)]
            shape_comp[0::2] = self.val_shape[:int((n_row ** 2) / 2)]
            del mask_val

            # comparison of scene
            scene_comp = torch.empty_like(self.latest_person)
            scene_comp[1::2] = gen_scene_val[:int((n_row ** 2) / 2)]
            scene_comp[0::2] = self.val_scene[:int((n_row ** 2) / 2)]
            del gen_scene_val

            grid_person = make_grid(person_comp, nrow=n_row)
            save_image(grid_person, gen_person_file_val)
            del person_comp
            grid_shape = make_grid(shape_comp, nrow=n_row)
            save_image(grid_shape, shape_file_val)
            del shape_comp
            grid_scene = make_grid(scene_comp, nrow=n_row)
            save_image(grid_scene, scene_file_val)
            del scene_comp

            # interpolations between samples of specified images
            self.save_interpolation(current_depth, epoch, from_train=False, batch_num=i, draw_latent=False, alpha=alpha,
                                    n_samples=self.n_gpus - 1)
            # interpolate between samples drawn from (0,1) gauss
            self.save_interpolation(current_depth, epoch, from_train=False, batch_num=i, draw_latent=True, alpha=alpha,
                                    n_samples=self.n_gpus - 1)
            # save generated images on background without person
            self.generate_no_bg_person(current_depth, epoch, i, alpha=alpha)

        self.save_interpolation(current_depth, epoch, from_train=True, batch_num=i, draw_latent=False, alpha=alpha,
                                n_samples=self.n_gpus - 1)
        self.save_interpolation(current_depth, epoch, from_train=True, batch_num=i, draw_latent=True, alpha=alpha,
                                n_samples=self.n_gpus - 1)

        self.generator.train()

    def generate_no_bg_person(self, depth, epoch, iteration, alpha):

        to_tensor = ToTensor()
        batch = self.val_fix_batch
        tgt_size = self.smallest_size * (2 ** depth)
        downsampler = lambda x: F.interpolate(x, tgt_size, mode='bilinear')

        # load images
        bg_dir = './data/backgrounds'
        background = [os.path.join(bg_dir, img) for img in os.listdir(bg_dir) if '.py' not in img and 'new' not in img]
        background = torch.stack([to_tensor(Image.open(img).convert('RGB')) for img in background]).to(self.device)
        n_bg = min([len(background), len(self.val_fix_batch['image'])])
        n_bg = max([n_bg // self.n_gpus, 1]) * self.n_gpus
        while n_bg > len(background):
            background = torch.cat((background, background))
        idx_from = random.randint(0, len(background) - n_bg)
        background = background[idx_from:(idx_from + n_bg)]
        background = downsampler(background)

        # get keypoints
        keypoints = batch['mask_keypoint'][:n_bg].to(self.device)
        keypoints = downsampler(keypoints)
        # create masks
        with torch.no_grad():
            estimated_masks = self.estimate_mask(self.mask_estimator, keypoints)
            estimated_masks = downsampler(estimated_masks)

        # take persons from original images, mask them, and obtain appearance vectors
        images = downsampler(batch['image'][:n_bg].to(self.device))
        persons = images * estimated_masks
        with torch.no_grad():
            z = self.encode_z(persons)

        # generate person image
        masked_bg = background * (1 - estimated_masks)
        with torch.no_grad():
            gen_input = torch.cat((masked_bg, keypoints), dim=1)
            gen_output = self.generator(gen_input, z, depth, alpha)
            gen_scene = gen_output * estimated_masks + masked_bg

        skeletons = torch.sum(keypoints, dim=1).unsqueeze(1).clamp_(0, 1).repeat([1, 3, 1, 1])
        estimated_masks = estimated_masks.repeat([1, 3, 1, 1])
        samples = [
            background, skeletons, estimated_masks, masked_bg, persons, gen_output, gen_scene
        ]
        samples = torch.cat(samples, dim=3)

        grid = make_grid(samples, nrow=1, normalize=True)
        save_image(grid, os.path.join(self.args.sample_dir, 'free_bg_{}_{}_b{}.jpg'.format(depth, epoch, iteration)))

        del background
        del keypoints
        del images
        del persons
        del z
        del masked_bg
        del gen_output
        del gen_scene
        del skeletons
        del estimated_masks
        del samples
        del grid

    def sample_latent(self, size, mu=0, std=1.0):
        z = torch.Tensor(*size).data.normal_(mu, std)
        return z

    def create_input(self, batch):
        bs = batch['image'].size(0)  # batch size
        inp = {'data': [], 'encoder': None}
        for name in self.cond_input:
            if name == 'encoder':
                enc_in = (batch['image'] * batch['mask_orig']).to(self.args.device)
                enc_out, _, _ = self.encode_z(enc_in)
                side = int(np.sqrt(enc_out.shape[-1]))
                enc_out = enc_out.view(bs, 1, side, side)
                inp['encoder'] = enc_out
            else:
                tmp_data = batch[name].to(self.args.device)
                if len(tmp_data.shape) != 4:
                    tmp_data = tmp_data.unsqueeze(0)
                inp['data'].append(tmp_data)
        inp['data'] = torch.cat(inp['data'], dim=1).to(self.args.device)
        return inp

    def save_interpolation(self, depth, epoch, from_train, batch_num, n_samples=10, draw_latent=False, alpha=None):
        size = [1, self.args.z_dim]
        tgt_size = self.smallest_size * (2 ** depth)
        downsampler = lambda x: F.interpolate(x, tgt_size, mode='bilinear')

        batch = self.trn_fix_batch if from_train else self.val_fix_batch
        images = batch['image']
        kpts = batch['mask_keypoint'].to(self.device)
        kpts = downsampler(kpts)
        with torch.no_grad():
            estimated_masks = self.estimate_mask(self.mask_estimator, kpts)
            estimated_masks = F.interpolate(estimated_masks, (tgt_size, tgt_size), mode='bilinear')
            mask_from = estimated_masks[0].unsqueeze(0)
        images = F.interpolate(images, estimated_masks.shape[-2:]).to(self.device)
        persons = (images * estimated_masks).to(self.device)
        person_from = persons[0].unsqueeze(0)
        person_to = persons[1].unsqueeze(0)

        if draw_latent:
            z_from = self.sample_latent(size).to(self.args.device)
            z_to = self.sample_latent(size).to(self.args.device)
            person_from, person_to = None, None
        else:
            with torch.no_grad():
                # self.appearance_encoder.eval()
                z_all = self.encode_z(persons)
                z_from = z_all[0].unsqueeze(0)
                z_to = z_all[1].unsqueeze(0)

        # create input appearance vectors
        delta = (z_to - z_from) / n_samples
        zs = torch.cat([z_from + i * delta for i in range(n_samples + 1)]).to(self.args.device)

        # create conditional input for the generator = masked image + keypoints
        image = downsampler(batch['image'][0].unsqueeze(0))
        bg = (image.to(self.device) * (1 - mask_from)).to(self.args.device)

        # downsampled versions
        orig_mask_dwn = downsampler(batch['mask_orig'][0].unsqueeze(0)).repeat([1, 3, 1, 1]).to(self.device)
        skeleton = downsampler(
            torch.sum(kpts, dim=1).unsqueeze(1).repeat([1, 3, 1, 1]).to(self.args.device))
        orig_img = images[0].unsqueeze(0).repeat([zs.shape[0], 1, 1, 1]).to(self.device)
        kpts_from = kpts[0].unsqueeze(0).repeat([zs.shape[0], 1, 1, 1]).to(self.device)

        with torch.no_grad():
            # get generated image
            masked_images = orig_img * (1 - mask_from)
            cond_input = torch.cat((masked_images, kpts_from), dim=1)
            gen_output = self.generator(cond_input, zs, depth, alpha=alpha)
            gen_scene = gen_output * mask_from + orig_img * (1 - mask_from)

        zeros = torch.zeros(1, 3, tgt_size, tgt_size).to(self.args.device)
        masked_image = masked_images[0].unsqueeze(0)
        mask_from = mask_from.repeat([len(gen_output), 3, 1, 1])
        orig_img = orig_img[0].unsqueeze(0)
        skeleton = skeleton[0].unsqueeze(0)
        if draw_latent:  # do not draw persons
            stacked = torch.cat([orig_img, masked_image, skeleton, gen_scene, \
                                 zeros, skeleton, orig_mask_dwn, mask_from])
            nrow = 3 + gen_scene.shape[0]
        else:
            person_from = downsampler(person_from)
            person_to = downsampler(person_to)
            stacked = torch.cat([orig_img, masked_image, skeleton, person_from, gen_output, person_to, \
                                 orig_img, masked_image, skeleton, person_from, gen_scene, person_to, \
                                 zeros, zeros, skeleton, orig_mask_dwn, mask_from, zeros])
            nrow = 5 + gen_scene.shape[0]

        # create grid and save it
        latent_str = '_latent' if draw_latent else ''
        split = 'trn' if from_train else 'val'
        fname = os.path.join(self.args.sample_dir, "interp_{}{}_".format(split, latent_str) + str(depth) +
                             "_" + str(epoch) + '_b{}'.format(batch_num) + ".jpg")
        grid = make_grid(stacked, nrow=nrow)
        save_image(grid, fname)

        del kpts
        del images
        del persons
        del orig_img
        del kpts_from
        del orig_mask_dwn
        del person_from
        del person_to
        del z_from
        del z_to
        del bg
        del skeleton
        del zeros
        del zs

    def train(self, train_set, val_set):

        # create a global time counter
        global_time = time.time()

        iter_ticker = 0
        for current_depth in range(self.args.start_depth, self.args.depth):
            torch.cuda.empty_cache()

            if not self.args.no_cudnn:
                print('set backends.cudnn.benchmark = True')
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.enabled = False
                torch.backends.cudnn.benchmark = True  # boost the speed
                torch.backends.cudnn.enabled = True

            print("\n\nCurrently working on Depth: ", current_depth)
            current_res = self.args.image_size / np.power(2, self.args.depth - current_depth - 1)
            print("Current resolution: %d x %d" % (current_res, current_res))

            # optionally set batch_sampler=None
            cur_bs = self.args.batch_sizes[current_depth]
            bs_per_gpu = cur_bs // torch.cuda.device_count()
            num_workers = min([bs_per_gpu, self.args.num_workers])
            print('Batch size: {} ({} per GPU) => num_workers={}'.format(cur_bs, bs_per_gpu, num_workers))
            train_loader = get_data_loader(train_set, num_workers=bs_per_gpu,
                                           batch_size=cur_bs,
                                           seed=self.args.seed, pin_memory=self.args.pin_memory,
                                           shuffle=self.args.shuffle, drop_last=True)
            fixed_train = None

            if val_set is not None:
                self.use_val = True
                val_loader = get_data_loader(val_set, self.args.batch_sizes[current_depth], self.args.num_workers,
                                             seed=self.args.seed, pin_memory=self.args.pin_memory)
                fixed_val = next(iter(val_loader))
                idx_val = np.random.randint(len(val_set), size=2)
                self.val_keypoints = self.__progressive_downsampling(fixed_val['mask_keypoint'], current_depth)
                with torch.no_grad():
                    estimated_mask_full = self.estimate_mask(self.mask_estimator, self.val_keypoints)
                    estimated_mask = F.interpolate(estimated_mask_full, self.val_keypoints.shape[-2:], mode='bilinear')
                    estimated_mask_full = F.interpolate(estimated_mask_full, fixed_val['image'].shape[-2:],
                                                        mode='bilinear').cpu()
                self.val_fix_batch = self.sample_batch(fixed_val, max([16, self.n_gpus]))
                fixed_val['mask_orig'] = estimated_mask.cpu()
                self.val_scene = self.__progressive_downsampling(fixed_val['image'], current_depth)
                fixed_val['image_masked'] = self.val_scene * (1 - fixed_val['mask_orig'])
                self.val_person_full = fixed_val['image'] * estimated_mask_full
                self.val_person = self.val_scene * fixed_val['mask_orig']
                self.val_shape = fixed_val['mask_orig']
                self.val_background = fixed_val['image_masked']
                masked_images = self.val_scene * estimated_mask.cpu()
                self.val_cond = torch.cat((masked_images, self.val_keypoints), dim=1)
            else:
                self.use_val = False
                val_loader = None

            self.trn_fix_batch = None

            total_batches = len(train_loader)

            # when will the fading stop
            fader_point = int((self.args.fade_in_percentage[current_depth] / 100)
                              * self.args.epochs[current_depth] * total_batches)
            print("Fade new layer for {} iterations.".format(fader_point))

            ticker = 1

            for epoch in range(1, self.args.epochs[current_depth] + 1):
                start = timeit.default_timer()  # record time at the start of epoch

                print("\nEpoch: {}".format(epoch))

                idx_trn = np.random.randint(len(train_set), size=2)

                if self.args.retrieve_nn:
                    self.trn_fix_batch_from = self.trn_fix_batch_from[0]
                    self.trn_fix_batch_to = self.trn_fix_batch_to[0]

                step = 0  # counter for number of iterations

                load_time_epoch = 0
                load_start = time.time()
                last_feedback_iter = 0
                load_t_buff = 0
                n_iters_wo_feedback = 0

                for (i, batch) in enumerate(train_loader, 1):
                    load_t_buff += time.time() - load_start
                    n_iters_wo_feedback += 1
                    provide_feedback = (self.args.feedback_factor == -1) or (i == 1) or \
                                       (i % max([1, int(total_batches / self.args.feedback_factor)]) == 0)

                    if provide_feedback:
                        load_time_epoch += load_t_buff
                        load_time_avg = load_t_buff / n_iters_wo_feedback
                        n_iters_wo_feedback = 0
                        load_t_buff = 0

                    if self.args.retrieve_nn:
                        batch, batch_nn = batch
                    if self.trn_fix_batch is None:
                        self.trn_fix_batch = self.sample_batch(batch, max([2, self.n_gpus]))
                    iter_ticker += 1
                    # calculate the alpha for fading in the layers
                    alpha = ticker / fader_point if ticker <= fader_point else 1

                    if fixed_train is None:
                        fixed_train = batch

                    # prepare the input
                    # cond_input_train = self.create_input(batch)
                    # background = self.__progressive_downsampling(batch['image_masked'], current_depth)
                    self.target_mask_full = batch['mask_orig'].to(self.device)
                    self.target_mask = self.__progressive_downsampling(batch['mask_orig'], current_depth).to(
                        self.device)
                    self.keypoints_full = batch['mask_keypoint'].to(self.device)
                    self.keypoints = self.__progressive_downsampling(batch['mask_keypoint'], current_depth).to(
                        self.device)

                    # take original
                    self.target_person_full = (batch['image'] * batch['mask_orig']).to(self.device)
                    self.target_person = self.__progressive_downsampling(self.target_person_full,
                                                                         current_depth).to(self.device)
                    scene = batch['image']
                    self.target_scene = self.__progressive_downsampling(scene, current_depth).to(self.device)

                    # run one training step
                    self.train_one_step(self.keypoints, self.target_scene,
                                        self.target_mask, current_depth, alpha)

                    # provide a loss feedback
                    if provide_feedback:
                        elapsed = time.time() - global_time
                        elapsed = str(datetime.timedelta(seconds=elapsed))

                        mem = '' if 'cuda' not in self.device else ',mem: {:.2f}G'.format(
                            torch.cuda.memory_cached() / 1E9)

                        dis_loss = sum([l for l in self.latest_losses['dis'].values()])
                        gen_loss = sum([l for l in self.latest_losses['gen'].values()])
                        dis_loss_str = "; ".join(
                            ['{}: {:.3f}'.format(name, val) for name, val in self.latest_losses['dis'].items()])
                        gen_loss_str = "; ".join(
                            ['{}: {:.3f}'.format(name, val) for name, val in self.latest_losses['gen'].items()])
                        print("Elapsed: [{}]  batch: {:d}/{:d} (loading: {:.2f});  d_loss: {:.3f} ({});  g_loss: {:.3f}"
                              " ({}){}, alpha={:.3f}".format(elapsed, i, total_batches, load_time_avg, dis_loss,
                                                             dis_loss_str, gen_loss, gen_loss_str, mem, alpha))

                        # also write the losses to the log file:
                        os.makedirs(self.args.log_dir, exist_ok=True)
                        log_file = os.path.join(self.args.log_dir, "loss_" + str(current_depth) + ".log")
                        with open(log_file, "a") as log:
                            log.write(str(step) + "\t" + str(dis_loss) +
                                      "\t" + str(gen_loss) + "\n")

                        # create a grid of samples and save it
                        os.makedirs(self.args.sample_dir, exist_ok=True)

                        # save example images
                        if self.args.plot_factor == -1 or (epoch % self.args.plot_factor == 0) and (i == 1):
                            self.save_results(i, current_depth, epoch, alpha)

                        # save to tensorboardX writer
                        for name, val in self.latest_losses['gen'].items():
                            self.writer.add_scalar('gen_' + name, val, iter_ticker)
                        for name, val in self.latest_losses['dis'].items():
                            self.writer.add_scalar('dis_' + name, val, iter_ticker)
                            # for dataset_name, val in self.latest_losses['dis_scene'].items():
                            #    self.writer.add_scalar('dis_scene_' + dataset_name, val, iter_ticker)

                    # increment the alpha ticker and the step, set start time of the loading
                    ticker += 1
                    step += 1
                    load_start = time.time()

                    del self.keypoints
                    del self.target_mask_full
                    del self.target_mask
                    del self.keypoints_full
                    del self.target_person_full
                    del self.target_person
                    del self.target_scene

                    # save current state of network
                if epoch % self.args.checkpoint_factor == 0 or (epoch == self.args.epochs[current_depth] and epoch > 3):
                    os.makedirs(self.args.save_dir, exist_ok=True)
                    gen_save_file = os.path.join(self.args.save_dir, "GAN_GEN_" + str(current_depth) + ".pth")
                    dis_save_file = os.path.join(self.args.save_dir,
                                                 "GAN_DIS_SCENE_" + str(current_depth) + ".pth")
                    blending_save_file = os.path.join(self.args.save_dir,
                                                      "BLENDING_" + str(current_depth) + ".pth")
                    encoder_save_file = os.path.join(self.args.save_dir,
                                                     "ENCODER_" + str(current_depth) + ".pth")
                    gen_optim_save_file = os.path.join(self.args.save_dir,
                                                       "GAN_GEN_OPTIM_" + str(current_depth) + ".pth")
                    dis_optim_save_file = os.path.join(self.args.save_dir,
                                                       "GAN_DIS_SCENE_OPTIM_" + str(current_depth) + ".pth")

                    # save networks
                    torch.save(self.generator.state_dict(), gen_save_file)
                    torch.save(self.discriminator_scene.state_dict(), dis_save_file)
                    torch.save(self.mask_estimator.state_dict(), blending_save_file)
                    torch.save(self.appearance_encoder.state_dict(), encoder_save_file)

                    # save optimizers
                    torch.save(self.optim_G.state_dict(), gen_optim_save_file)
                    torch.save(self.optim_D.state_dict(), dis_optim_save_file)

                    # also save the shadow generator if use_ema is True
                    if self.args.use_ema:
                        gen_shadow_save_file = os.path.join(self.args.save_dir, "GAN_GEN_SHADOW_" +
                                                            str(current_depth) + ".pth")
                        torch.save(self.gen_shadow.state_dict(), gen_shadow_save_file)

                stop = timeit.default_timer()
                print("Time taken for epoch: {:.1f} secs ({:.1f}s loadining)".format(stop - start, load_time_epoch))

    def estimate_mask(self, estimator, keypoints, thr=0.75):
        keypoints_me = self.me_input_resizer(keypoints.to(self.device))
        keypoints = keypoints.to('cpu')
        estimated_mask, _ = estimator(keypoints_me)
        estimated_mask = estimated_mask.clamp_(0, 1)
        estimated_mask[estimated_mask > thr] = 1.0
        estimated_mask[estimated_mask <= thr] /= thr
        return estimated_mask
