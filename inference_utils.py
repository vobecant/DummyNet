import json
import os
from collections import OrderedDict

import torch
import torch.nn.functional as F

from models.DummyGAN import DummyGenerator
from models.MaskEstimator import MaskEstimator
from models.networks import VAE_enc_only


def load_networks(saves_path):
    args_file = None
    for fname in os.listdir(saves_path):
        if '_opts.txt' in fname:
            args_file = os.path.join(saves_path, fname)
            break

    assert args_file

    args = OrderedDict()
    with open(args_file, 'r') as f:
        args.__dict__ = json.load(f)

    me_chkpt = os.path.join(saves_path, "MASK_ESTIMATOR.pth")
    me_chkpt = torch.load(me_chkpt, map_location=lambda storage, loc: storage)
    me_args = me_chkpt['args']

    args.save_dir = '{}{}'.format(args.save_dir, args.name)
    last_depth = int(args.depth) - 1

    gen_save_file = os.path.join(saves_path, "GAN_GEN_{}.pth".format(last_depth))
    mask_estimator_save_file = os.path.join(saves_path, "BLENDING_{}.pth".format(last_depth))
    encoder_save_file = os.path.join(saves_path, "ENCODER_{}.pth".format(last_depth))

    generator = DummyGenerator(args)
    weights = torch.load(gen_save_file, map_location=lambda storage, loc: storage)
    w = OrderedDict()
    for k, v in weights.items():
        name = k.replace('module.', '')
        w[name] = v
    generator.load_state_dict(w)

    appearance_encoder = VAE_enc_only(in_channels=3, code_dim_base=args.z_dim)
    weights = torch.load(encoder_save_file, map_location=lambda storage, loc: storage)
    w = OrderedDict()
    for k, v in weights.items():
        name = k.replace('module.', '')
        w[name] = v
    appearance_encoder.load_state_dict(w)

    mask_estimator = MaskEstimator(me_args, 17)
    weights = torch.load(mask_estimator_save_file, map_location=lambda storage, loc: storage)
    w = OrderedDict()
    for k, v in weights.items():
        name = k.replace('module.', '')
        w[name] = v
    mask_estimator.load_state_dict(w)

    return args, generator, appearance_encoder, mask_estimator


def encode_appearance(appearance_encoder, real_person, encoder_input_size):
    with torch.no_grad():
        real_person_resized = F.interpolate(real_person, encoder_input_size, mode='bilinear')
        z = appearance_encoder(real_person_resized, return_mu_logvar=False)
    return z


def estimate_mask(estimator, me_input_resizer, keypoints, thr=0.75):
    keypoints_me = me_input_resizer(keypoints)
    keypoints = keypoints.to('cpu')
    # keypoints = keypoints.to('cuda')
    estimated_mask, _ = estimator(keypoints_me)
    # estimated_mask, _ = estimator(keypoints_me.to('cuda'))
    estimated_mask = estimated_mask.clamp_(0, 1)
    estimated_mask[estimated_mask > thr] = 1.0
    estimated_mask[estimated_mask <= thr] /= thr
    return estimated_mask
