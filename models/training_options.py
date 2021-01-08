import argparse
import json
import os
import zipfile

from models.utils import *


class BaseOptions():
    def __init__(self, load_file=None, save=True):
        self.save = save
        self.parser = argparse.ArgumentParser()
        if load_file is not None:
            self.args = self.parser.parse_args()
            with open(load_file, 'r') as f:
                self.args.__dict__ = json.load(f)
            self.args.eval = True
        else:
            self.initialize(self.parser)
            self.args = self.parser.parse_args()
            self.parse()

    def get_args(self):
        return self.args

    def initialize(self, parser):

        # general
        parser.add_argument('--name', type=str, default=None, help='Name of the experiment.')
        parser.add_argument('--num_training_samples', type=int, default=-1, help='Maximal number of training samples.')
        parser.add_argument('--num_samples', type=int, default=32, help='number of samples generated in sample_sheet')
        parser.add_argument('--eval', action='store_true')
        parser.add_argument('--multi_gpu', default=True)
        parser.add_argument('--single_gpu', action='store_true')
        parser.add_argument('--precompute_masks', action='store_true')
        parser.add_argument('--no_cudnn', action='store_true')
        parser.add_argument('--resume', action='store_true')

        # important
        parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for reading the data.')
        parser.add_argument('--D_type', type=str, default='multi',  # 'one',
                            help='Number of output values of the discriminator. Choose from: [one, multi].')
        parser.add_argument('--base_coco', type=str,
                            # default='/mnt/nas/data/MS_COCO',
                            # 'C:\\Users\\Antonin-PC\\school\\Master\\diploma thesis\\datasets\\coco',
                            # '/home/tonda/diploma_thesis/datasets/coco',
                            # '/mnt/nas/data/MS_COCO'
                            default='/home/vobecant/datasets/coco',
                            help='Path to the base of COCO folder.')
        parser.add_argument('--ybb_path',
                            # default='/mnt/nas/data/GoogleBoundingBoxes/videos/yt_bb_detection_train',
                            default='/home/vobecant/datasets/YBB/GAN_data/images',
                            help='Path to the base folder of YBB dataset split.')
        parser.add_argument('--ybb_precomputed_train',
                            default='./dataset/YBB/precomputed_idx/precomputed_idx_gan_train_crop256_sk190.json',
                            help='Path to precomputed indices for YBB dataset.')
        parser.add_argument('--ybb_precomputed_test',
                            default='./dataset/YBB/precomputed_idx/precomputed_idx_gan_test_crop256_sk190.json',
                            help='Path to precomputed indices for YBB dataset.')
        parser.add_argument('--ybb_split_json_train',
                            # default='/mnt/nas/data/GoogleBoundingBoxes/splits/first_debug/gan_train.json',
                            default='/home/vobecant/datasets/YBB/GAN_data/splits/gan_train.json',
                            help='JSON files with samples from given data split.')
        parser.add_argument('--ybb_split_json_test',
                            # default='/mnt/nas/data/GoogleBoundingBoxes/splits/first_debug/gan_train.json',
                            default='/home/vobecant/datasets/YBB/GAN_data/splits/gan_test.json',
                            help='JSON files with samples from given data split.')
        parser.add_argument('--ybb_masks_dir',
                            default='/home/vobecant/datasets/YBB/GAN_data/masks',
                            help='Path to the directory where are stored segmentations of people by Mask RCNN (joined with mask estimation from keypoints).')
        parser.add_argument('--coco_splits', nargs='*', default=['train2017'],
                            help='Splits of COCO dataset to be used during training.')
        parser.add_argument('--start_depth', type=int, default=0, help='Start training from this depth.')
        parser.add_argument('--epochs', type=int, nargs='*', default=[1, 1, 1, 1, 1],  # [300, 300, 400, 400, 1],
                            # [90, 90, 90],[300, 200, 200]
                            help='list of number of epochs to train the network for every resolution')
        parser.add_argument('--batch_sizes', type=int, nargs='*', default=[81, 81, 64, 32, 16],
                            # [100, 121, 36, 9, 4],  # [841, 625, 289, 72],
                            # [256, 121, 36, 9],  # [900, 400, 121, 25],[180,150,105,50]
                            # [400, 144, 49],[900, 400, 121]
                            help='list of batch_sizes for every resolution')
        parser.add_argument('--losses_gen', type=str,
                            default=[['l1_masked_avg_bg', 'l2_masked_avg_fg'],  # , 'edge_l1_soft'
                                     ['l1_masked_avg_bg', 'l2_masked_avg_fg'],  # , 'edge_l1_soft'
                                     ['l1_masked_avg_bg', 'l2_masked_avg_fg'],  # , 'edge_l1_soft'
                                     ['l1_masked_avg_bg', 'l2_masked_avg_fg']],  # , 'edge_l1_soft'
                            # , 'hist_masked_emd'],
                            nargs='*', help='Losses used for generator in addition to GAN loss.')

        # model
        parser.add_argument('--core_img_size', type=str, default=8,
                            help='Size of the image inputed to the core (smallest resolution).')
        parser.add_argument('--c_in', type=int, default=128, help='Number of input channels to the core module.')
        parser.add_argument('--c_out', type=int, default=128, help='Number of output channels from the core module.')
        parser.add_argument('--depth', type=int, default=5, help='Depth of down- and up-sampling stream.')
        parser.add_argument('--use_eql', type=bool, default=True, help='Whether to use equalized learning rate.')
        parser.add_argument('--use_gated', type=bool, default=True, help='Whether to use gated convolutions.')
        parser.add_argument('--gated_activation', type=str, default='lrelu',
                            help='Activation in the gated convolutions.')
        parser.add_argument('--core_downsamplings', type=int, default=1,
                            help='Number of downsamplings (upsamplings) in core block.')
        parser.add_argument('--core_res_blocks', type=int, default=5, help='Number of residual blocks in core block.')
        parser.add_argument('--core_conv_blocks', type=int, default=2,
                            help='Number of convolutional blocks after upsampling in the core block.')

        # GAN setup
        parser.add_argument('--use_ema', action='store_true', help='Whether to use Exponential Moving Averages.')
        parser.add_argument('--ema_decay', type=float, default=0.999, help='Value of mu for ema.')
        parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                            help='Device to run the GAN on (GPU~cuda / CPU~cpu).')
        parser.add_argument('--n_critic', type=int, default=1,
                            help='Number of times to update discriminator per generator update.')

        # optimizer setup
        parser.add_argument('--drift', type=float, default=0.001,
                            help='Drift penalty for the loss (Used only if loss is wgan or wgan-gp).')
        parser.add_argument('--beta_1', type=float, default=0, help='beta_1 for Adam')
        parser.add_argument('--beta_2', type=float, default=0.99, help='beta_2 for Adam')
        parser.add_argument('--eps', type=float, default=1e-8, help='epsilon for Adam')

        # losses setup
        parser.add_argument('--loss', type=str, default=['wgan-gp_masked'], nargs='*',
                            help='Name of the loss function to be used.')
        parser.add_argument('--dis_use_no_kpts', action='store_true',
                            help='Do not use keypoints at the input of the discriminator.')
        parser.add_argument('--gan_loss_person_only', action='store_true')
        parser.add_argument('--use_luminance_loss', action='store_true')
        parser.add_argument('--luminance_loss_lbd', type=float, default=1.0)
        parser.add_argument('--n_bins', type=int, default=16, help='Number of histogram bins.')
        parser.add_argument('--use_kl_style', action='store_true', help='Use KL loss to enforce N(0,1) of style codes.')
        parser.set_defaults(use_kl_style=True)
        # feature matching losses
        parser.add_argument('--use_fm_loss', action='store_true')
        parser.set_defaults(use_fm_loss=False)
        parser.add_argument('--fm_lbd', type=float, default=10.0)
        parser.add_argument('--use_vgg_loss', action='store_true')
        parser.set_defaults(use_vgg_loss=False)
        parser.add_argument('--vgg_lbd', type=float, default=10.0)

        # training setup
        parser.add_argument('--learning_rate', type=float, default=5e-4, help='Learning rate for Adam.')
        parser.add_argument('--fade_in_percentage', type=float, default=[20, 20, 20, 20, 20], nargs='*',
                            help='List of percentages of epochs per resolution. Used for fading in the new layer. '
                                 'Not used for first resolution, but dummy value still needed.')
        parser.add_argument('--feedback_factor', type=int, default=100, help='Number of logs per epoch.')
        parser.add_argument('--plot_factor', type=int, default=-1,
                            help='Number of plots per epoch. -1 means that the results are plotted every time the feedback is shown.')
        parser.add_argument('--checkpoint_factor', type=int, default=5,
                            help='Save model after these many epochs. Note that only one model is stored '
                                 'per resolution. During one resolution, the checkpoint will be updated (rewritten) '
                                 'according to this factor.')
        parser.add_argument('--seed', type=int, default=42, help='Initial seed of random generators.')
        parser.add_argument('--save_dir', type=str, default='./trained_models/',
                            help='Directory for saving the models (.pth files).')

        # files and directories
        parser.add_argument('--log_dir', type=str, default='./logs/', help='Directory for saving the loss logs.')
        parser.add_argument('--sample_dir', type=str, default='./samples/',
                            help='Directory for saving the generated examples.')
        parser.add_argument('--files_to_save', nargs='?',
                            default=['./models/customLayers.py', './models/DataTools.py',
                                     './models/Losses.py', './models/networks.py',
                                     './models/training_options.py', './models/utils.py', './dataloader.py',
                                     './loaders.py', './main_dummy_gan.py', './models/DummyGAN.py'],
                            help='Files that will be compressed and saved.')

        # generator setup
        parser.add_argument('--learn_background', action='store_true', \
                            help='If true, network learns the background as well. Otherwise it \
                just inserts generated image in the mask to the original image.')
        parser.set_defaults(learn_background=False)
        parser.add_argument('--learn_blending', action='store_true',
                            help='Use a separate network for deciding which pixels keep from generated and which from original image.')
        parser.set_defaults(learn_blending=False)
        parser.add_argument('--use_nn', action='store_true', help='Use nearest neighbor during training.')
        parser.set_defaults(use_nn=False)
        parser.add_argument('--use_style', action='store_true', help='Use style encoding.')
        parser.set_defaults(use_style=True)
        parser.add_argument('--style_dim', type=int, default=8, help='Style dimension.')

        # image encoder setup
        parser.add_argument('--use_pretrained_encoder', action='store_true',
                            help='Use weights of pretrained encoder by VAE.')
        parser.set_defaults(use_pretrained_encoder=True)
        parser.add_argument('--encoder_input_size', default=64, help='Size of image input to the encoder.')

        # mask estimator setup
        parser.add_argument('--use_pretrained_mask_estimator', action='store_true')
        parser.set_defaults(use_pretrained_mask_estimator=True)

        # style GAN setup
        parser.add_argument('--style_gen_latent', type=int, default=64,
                            help='Size of input latent vector to style generator.')
        parser.add_argument('--style_gen_hidden', type=int, default=64,
                            help='Size of hidden layers in style generator.')
        parser.add_argument('--style_dis_hidden', type=int, default=64,
                            help='Size of hidden layers in style discriminator.')
        parser.add_argument('--style_gen_layers', type=int, default=3, help='Number of layers in style generator.')
        parser.add_argument('--style_dis_layers', type=int, default=3, help='Number of layers in style discriminator.')
        parser.add_argument('--style_gan_loss', type=str, default='wgan-gp_style', help='Loss for style GAN.')
        parser.add_argument('--nn_prob', type=float, default=0.0,
                            help='Probability of taking the style from the nearest neighbor instead of from the ground truth.')

        # dataset setup
        parser.add_argument('--retrieve_nn', action='store_true', help='Return the nearest neighbor.')
        parser.add_argument('--precomputed_nn', default='./trained_models/precomputed_nn.npy',
                            help='Path to the precomputed LUT of NN.')
        parser.add_argument('--val_sets', nargs='*', default=['val2017'], help='Validation set.')
        parser.add_argument('--base_cityscapes', type=str, default='',  # '/args/datasets/cityscapes'
                            help='Path to the base of Cityscapes folder.')
        parser.add_argument('--batch_size', type=int, default=1, help='Batch size.')
        parser.add_argument('--shuffle', action='store_true', help='Shuffle the dataset.')
        parser.set_defaults(shuffle=True)
        parser.add_argument('--no_pin_memory', action='store_false', dest='pin_memory',
                            help='Pin memory in train_loader.')
        parser.add_argument('--flip_probability', type=float, default=0.5,
                            help='Probability of flipping the image horizontally.')
        parser.add_argument('-grayscale_probability', type=float, default=0.0,
                            help='Probability of turning image to grayscale. If image is RGB, it copies grayscale image to all 3 channels.')

        # YBB dataset setup
        parser.add_argument('--ybb_min_conf', default=None, help='Min requested skeleton detection confidence.')
        parser.add_argument('--ybb_max_blur', default=None, help='Max possible blur.')
        parser.add_argument('--ybb_min_darkness', default=None, help='Min amount of darkness.')

        # discriminator setup

        # ROCK and PREDICTOR options
        parser.add_argument('--use_rock', action='store_true', help='If true, use ROCK block in the discriminator.')
        parser.add_argument('--no_rock', action='store_false', help='Do not use ROCK block in the discriminator.')
        parser.set_defaults(use_rock=False)
        parser.add_argument('--rock_fusion_type', type=str, default='sum',
                            help='Type of feature fusion in the ROCK block. Possible values: [sum, product].')
        parser.add_argument('--shape_loss', type=str, default='l1_none', help='Type of loss for predictor in ROCK.')
        parser.add_argument('--use_pred', action='store_true', help='Use head predictor in discriminator.')
        parser.add_argument('--no_pred', action='store_false', help='No head predictor in discriminator.')
        parser.set_defaults(use_pred=False)

        # nearest neighbor search arguments
        parser.add_argument('--nn_path', type=str, default='./trained_models/train2017_keypoints_res128_v2.pth',
                            help='Path to the file with precomputed values.')
        parser.add_argument('--nn_save_file', type=str,
                            default='./trained_models/train2017_keypoints_res128_quadric_v2.pth',
                            help='Path to the file where the computed quadrics will be saved.')
        parser.add_argument('--dynamic_nn', action='store_true', dest='dynamic_nn',
                            help='Alpha channel value is sampled randomly each time.')
        parser.add_argument('--static_nn', action='store_false', dest='dynamic_nn', help='Use static alpha channel.')
        parser.set_defaults(dynamic_nn=False)
        parser.add_argument('--static_nn_alpha', default=0.5, type=float,
                            help='Static value used in alpha channel. Used if dynamic_nn=False.')

        # input data specification
        parser.add_argument('--image_size', type=int, default=256, help='Size of input image.')
        parser.add_argument('--min_person_size', type=int, default=190,
                            help='Required minimal size (max of height and width) of person.')
        parser.add_argument('--n_input_channels', type=int, default=4,
                            help='Number of input channels of training data.')
        parser.add_argument('--use_shape', dest='use_shape', action='store_true', help='If true, use shape in input.')
        parser.add_argument('--no_shape', dest='use_shape', action='store_false', help='Do not use shape in the input.')
        parser.set_defaults(use_shape=False)
        parser.add_argument('--use_kpts', dest='use_kpts', action='store_true', help='If true, use keypoints in input.')
        parser.add_argument('--no_kpts', dest='use_shape', action='store_false', help='Do not use keypoints in input.')
        parser.set_defaults(use_kpts=True)
        parser.add_argument('--enc2spade', dest='enc2spade', action='store_true',
                            help='Use encoder output as the input to spade layer.')
        parser.set_defaults(enc2spade=False)

        # SPADE_GAN options
        parser.add_argument('--use_spade', action='store_true')
        parser.set_defaults(use_spade=True)
        parser.add_argument('--lr_gen', default=0.0001, type=float, help='LR of SPADE generator.')
        parser.add_argument('--lr_dis', default=0.0004, type=float, help='LR of the discriminator in SPADE_GAN.')
        parser.add_argument('--ngf', default=64, type=int)
        parser.add_argument('--ndf', default=64, type=int)
        parser.add_argument('--z_dim', default=16, type=int)  # 400 previously
        parser.add_argument('--use_vae', action='store_true')
        parser.set_defaults(use_vae=True)
        parser.add_argument('--norm_G', default='spadesyncbatch3x3', type=str)
        parser.add_argument('--norm_D', default='instance', type=str)
        parser.add_argument('--norm_E', default='instance', type=str)
        parser.add_argument('--param_free_norm_type', default='syncbatch', type=str)
        parser.add_argument('--semantic_nc', type=int, default=17)
        parser.add_argument('--rec_loss_lbd', type=float, default=10)
        parser.add_argument('--kld_lbd', type=float, default=0.005, help='Weight of the KLD loss.')
        parser.add_argument('--loss_G_spade', type=str, default=[], nargs='*')
        parser.add_argument('--use_bg', dest='use_bg', action='store_true', help='Use background in SPADE input.')
        parser.set_defaults(use_bg=True)
        parser.add_argument('--gan_lbd', type=float, default=100.0, help='Weight of the GAN loss.')
        parser.add_argument('--mask_loss', default=[], nargs='*')
        parser.add_argument('--mask_loss_lbd', type=float, default=5.0)
        # paths to weight files
        parser.add_argument('--load_weights', action='store_true')
        parser.add_argument('--spade_g_weights', type=str, default='./trained_models/exp_spade7/GAN_GEN_1.pth')
        parser.add_argument('--spade_d_weights', type=str, default='./trained_models/exp_spade7/GAN_DIS_SCENE_1.pth')
        parser.add_argument('--spade_blending_weights', type=str, default='./trained_models/exp_spade7/BLENDING_1.pth')
        parser.add_argument('--mask_estimator_chkpt', type=str,
                            default='./trained_models/mask_estimator_coco_64px_noOutputActivation_run2/weights_latest.pth')
        parser.add_argument('--encoder_weights', type=str,
                            default='./trained_models/vae_appearance_run2/encoder_weights.pth')
        # default='./results/vae_lr_decay/checkpoint/weights_latest.pth'
        parser.add_argument('--spade_g_optim_weights', type=str,
                            default='./trained_models/exp_spade7/GAN_GEN_OPTIM_1.pth')
        parser.add_argument('--spade_d_optim_weights', type=str,
                            default='./trained_models/exp_spade7/GAN_DIS_PERSON_OPTIM_1.pth')
        # generator losses
        parser.add_argument('--encoder_cycle_loss', action='store_true')
        parser.set_defaults(encoder_cycle_loss=True)
        parser.add_argument('--encoder_cycle_loss_lbd', type=float, default=10.0)

        return parser

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def save_settings(self):
        # save command line arguments
        save_dir = '{}{}'.format(self.args.save_dir, self.args.name)
        name = self.args.name
        os.makedirs(save_dir, exist_ok=True)
        f_opts = '{}/{}_opts.txt'.format(save_dir, name)
        with open(f_opts, 'w') as f:
            json.dump(self.args.__dict__, f, indent=2)
        self.args.files_to_save.append(f_opts)
        # save used files
        if self.args.files_to_save and not self.args.eval:
            with zipfile.ZipFile('{}/{}_files.zip'.format(save_dir, name), 'w') as zfile:
                for f in self.args.files_to_save:
                    zfile.write(f)

    def parse(self):

        self.print_options(self.args)
        if self.save:
            self.save_settings()
