import os
import random
import sys
import time

import numpy as np
from torch.utils.data import DataLoader

tmp = '/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-1])
sys.path.append(tmp)

import matplotlib

matplotlib.use('Agg')

from dataloader import YBB_Dataset
from models.DummyGAN import Dummy_GAN
from models.training_options import BaseOptions
from models.utils import count_trainable_parameters
import torch


def set_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':

    path = os.getcwd()
    print("Working directory: {}\n".format(path))

    opt_load_file = None
    options = BaseOptions(opt_load_file)
    args = options.get_args()
    set_seeds(args.seed)

    if not args.no_cudnn:
        print('set backends.cudnn.benchmark = True')
        torch.backends.cudnn.benchmark = True  # boost the speed
        torch.backends.cudnn.enabled = True

    if args.single_gpu:
        args.multi_gpu = False
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    cs_root = args.base_cityscapes
    transforms = None
    flip_probability = args.flip_probability
    batch_size = 1
    shuffle = False
    img_size = 256
    min_height = 190
    precomputed_nn = None
    precomputed_idx_dir = './dataset/YBB/precomputed_idx/eccv/2019-11-05_unique_vid/'

    # set parameters for nearest neighbor search if required
    nn_args = None

    model = Dummy_GAN(args, debug=False)
    G_params = count_trainable_parameters(model.generator)
    D_params = count_trainable_parameters(model.discriminator_scene)
    E_params = count_trainable_parameters(model.appearance_encoder)
    B_params = count_trainable_parameters(model.mask_estimator)
    print('__________________________________________')
    print('Number of trainable parameters:')
    print('\tgenerator:        {}'.format(G_params))
    print('\tdiscriminator:    {}'.format(D_params))
    print('\tencoder:          {}'.format(E_params))
    print('\tblending net:     {}'.format(B_params))
    print('__________________________________________')

    # YBB dataset
    ybb_path = args.ybb_path
    transformations = None
    crop_size = args.image_size
    min_sk_size = args.min_person_size
    req_joints_coco = [[0, 1, 2, 3, 4],  # at least one of 'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear'
                       [5, 6],  # at least one of 'left_shoulder', 'right_shoulder'
                       [7, 8],  # at least one of 'left_elbow', 'right_elbow'
                       [11, 12],  # at least one of 'left_hip', 'right_hip'
                       [13, 14]]  # at least one of 'left_knee', 'right_knee'
    overlap = 1.3
    scale_range = (1.0, 1.2)
    rot_max = 20  # in degrees
    flip_prob = args.flip_probability
    debug = False
    ybb_split_json_train = args.ybb_split_json_train
    _, json_train_split = os.path.split(ybb_split_json_train)
    json_train_split_name = json_train_split.split('.')[0]
    ybb_precomputed_train = args.ybb_precomputed_train

    ybb_split_json_test = args.ybb_split_json_test
    _, json_test_split = os.path.split(ybb_split_json_test)
    json_test_split_name = json_test_split.split('.')[0]
    ybb_precomputed_test = args.ybb_precomputed_test

    ybb_split_json_test = args.ybb_split_json_test
    annot_subfolder_name = None
    ybb_train_dataset = YBB_Dataset(ybb_path, transformations=transformations, crop_size=crop_size, overlap=overlap,
                                    scale_range=scale_range, rot_max=rot_max, flip_prob=flip_prob,
                                    min_sk_size=min_sk_size, required_joints_coco=req_joints_coco,
                                    min_conf=args.ybb_min_conf, max_blur=args.ybb_max_blur,
                                    min_darkness=args.ybb_min_darkness, debug=debug, split_json=ybb_split_json_train,
                                    precomputed_data_path=ybb_precomputed_train,
                                    annot_subfolder_name=annot_subfolder_name,
                                    masks_dir=args.ybb_masks_dir, images_dir=args.ybb_path)
    ybb_test_dataset = YBB_Dataset(ybb_path, transformations=transformations, crop_size=crop_size, overlap=overlap,
                                   scale_range=scale_range, rot_max=rot_max, flip_prob=flip_prob,
                                   min_sk_size=min_sk_size, required_joints_coco=req_joints_coco,
                                   min_conf=args.ybb_min_conf, max_blur=args.ybb_max_blur,
                                   min_darkness=args.ybb_min_darkness, debug=debug, split_json=ybb_split_json_test,
                                   precomputed_data_path=ybb_precomputed_test,
                                   annot_subfolder_name=annot_subfolder_name,
                                   masks_dir=args.ybb_masks_dir, images_dir=args.ybb_path)

    train_dataset = ybb_train_dataset  # JointDataset(coco_train_dataset, gbb_train_dataset)

    if args.num_training_samples > 0:
        print('Randomly choose {} samples for training.'.format(args.num_training_samples))
        former_idx = ybb_train_dataset.idx
        random.seed(args.seed)
        random.shuffle(former_idx)
        shuffled_cropped = former_idx[:args.num_training_samples]
        ybb_train_dataset.idx = shuffled_cropped

    print(
        'Dataset sizes:\n\ttraining:\t{}\n\tvalidation/testing:\t{}'.format(len(train_dataset), len(ybb_test_dataset)))

    start = time.time()
    if args.precompute_masks:
        bs = 8
        for dataset in [ybb_train_dataset, ybb_test_dataset]:  # , ybb_test_dataset
            loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4)
            for i, batch in enumerate(loader):
                if i % 100 == 0:
                    samples = (i + 1) * bs
                    print('batch {}/{}, samples: {}/{}, {:.1f}s'.format(i, len(loader), samples, len(dataset),
                                                                        time.time() - start))
                pass


    else:
        model.train(ybb_train_dataset, ybb_test_dataset)
