import math
import numbers
import os
import pickle
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
from torchvision.transforms import ToTensor
from torchvision.utils import save_image

from inference_utils import load_networks, encode_appearance, estimate_mask
from models.DummyGAN import KeypointsDownsampler
from position_proposer import PositionProposer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
assert device == 'cuda'


class GaussianSmoothing(torch.nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """

    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.weight = self.weight.to(device)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)


to_pil = ToPILImage()
BATCH_SIZE = 4
GEN_INPUT_SIZE = 256
MAX_SIZE_CROP = 300
OUTPUT_SIZE = 128
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
KW = 11
smoothing = GaussianSmoothing(3, KW, 1)

if __name__ == '__main__':
    nets_dir = sys.argv[1]
    save_dir = sys.argv[2]  # /home/vobecant/datasets/nightowls/generated_training_scenes
    bb_file = sys.argv[3]  # /home/vobecant/PhD/CSP/data/cache/nightowls/train_h50_nonempty_xyxy
    START_SAMPLE_ID = int(sys.argv[4]) if len(sys.argv) > 4 else -1
    END_SAMPLE_ID = int(sys.argv[5]) if len(sys.argv) > 5 else -1
    xyxy_bboxes = True
    no_cuda = len(sys.argv) > 4
    bb_file_dir, bb_file_name = os.path.split(bb_file)
    save_file_extended_bbs = os.path.join(bb_file_dir,
                                          '{}_1P_hard_resize{}-{}'.format(bb_file_name, START_SAMPLE_ID, END_SAMPLE_ID))

    with open(bb_file, 'rb') as f:
        bboxes_orig = pickle.load(f)

    background_paths = [sample['filepath'] for sample in bboxes_orig]

    bboxes_orig = {os.path.split(sample['filepath'])[1]: sample for sample in bboxes_orig}

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    device = 'cpu' if no_cuda else device

    # load arguments and networks
    args, generator, encoder, mask_estimator = load_networks(nets_dir)
    me_input_resizer = KeypointsDownsampler(tgt_size=64, mode='bilinear')
    generator.eval()
    generator.to(device)
    encoder.eval()
    encoder.to(device)
    mask_estimator.eval()
    mask_estimator.to(device)

    to_tensor = ToTensor()
    position_proposer = PositionProposer(256, xyxy_boxes=xyxy_bboxes, max_size=MAX_SIZE_CROP)
    image_size = args.image_size

    with open('./data/YBB/night_samples.pkl', 'rb') as f:
        dataset = pickle.load(f)
    loader = iter(DataLoader(dataset, batch_size=1, shuffle=False))

    # extended annotations
    bboxes_extended = []

    with torch.no_grad():

        start_idx = 0
        end_idx = BATCH_SIZE
        n_iters = len(background_paths) // end_idx
        report_after = n_iters // 100
        print('Total number of iterations: {}, report after every {}th iteration.'.format(n_iters, report_after))
        it = 0
        start = time.time()
        n_runs_loader = 0

        rng = [i for i in range(len(background_paths))] if END_SAMPLE_ID < 0 else [i for i in
                                                                                   range(START_SAMPLE_ID,
                                                                                         END_SAMPLE_ID)]

        for pid, image_path in enumerate(background_paths, 1):
            if pid not in rng:
                continue
            scene = to_tensor(Image.open(image_path).convert('RGB'))
            _, image_name = os.path.split(image_path)
            print('image {}/{} {}'.format(pid, len(background_paths), image_path))
            objects_orig = bboxes_orig[image_name]
            objects_all = objects_orig.copy()

            # set paths
            person_num = 1
            images_dir = os.path.join(save_dir, '{}P'.format(person_num), 'images')
            labels_dir = os.path.join(save_dir, '{}P'.format(person_num), 'labels')
            bboxes_dir = os.path.join(save_dir, '{}P'.format(person_num), 'bboxes')
            crops_dir = os.path.join(save_dir, '{}P'.format(person_num), 'crops')
            if not os.path.exists(images_dir):
                os.makedirs(images_dir)
                os.makedirs(labels_dir)
                os.makedirs(bboxes_dir)
                os.makedirs(crops_dir)

            fname_img = os.path.join(images_dir, image_name)
            if os.path.isfile(fname_img):
                continue
            fname_crop = os.path.join(crops_dir, image_name.replace('.png', '_{}P.png'.format(person_num)))
            fname_crop_params = fname_crop.replace('.png', '.txt')

            # TODO: get a proper sample
            try:
                conditioned_sample = next(loader)
            except StopIteration:
                loader = iter(DataLoader(dataset, batch_size=1, shuffle=False))
                conditioned_sample = next(loader)
            keypoints = conditioned_sample['mask_keypoint'].to(device)
            skeleton = torch.sum(keypoints, dim=(0, 1)).clamp_(0, 1).cpu().numpy()
            person = conditioned_sample['image'].to(device)
            sk_y, sk_x = np.where(skeleton)

            while len(sk_y) == 0 or len(sk_x) == 0:
                try:
                    conditioned_sample = next(loader)
                except StopIteration:
                    loader = iter(DataLoader(dataset, batch_size=1, shuffle=False))
                    conditioned_sample = next(loader)
                keypoints = conditioned_sample['mask_keypoint'].to(device)
                skeleton = torch.sum(keypoints, dim=(0, 1)).clamp_(0, 1).cpu().numpy()
                person = conditioned_sample['image'].to(device)
                sk_y, sk_x = np.where(skeleton)

            with torch.no_grad():
                proposal = position_proposer(scene, None, objects_orig, objects_all, skeleton, skeleton)
                if proposal is None:
                    print('Didnt find any possible placement. Do not augment.')
                    Image.fromarray(scene).save(fname_img)
                    continue

                image_crop, crop_size, crop_params, bbox, image_crop_orig = proposal

                # add new bbox to the existing ones
                objects_all['bboxes'] = np.append(objects_all['bboxes'], np.asarray(bbox).reshape(1, -1), axis=0)

                # get estimated mask
                mask = estimate_mask(mask_estimator, me_input_resizer, keypoints)
                mask_cs = F.interpolate(mask, keypoints.shape[-2:], mode='bilinear')

                # get conditional appearance
                masked_person = person * mask_cs
                masked_person_npy = np.asarray(to_pil(masked_person[0].cpu()))
                z = encode_appearance(encoder, masked_person, 64)

                bg = to_tensor(image_crop).unsqueeze(0).to(device)
                bg_orig = to_tensor(image_crop_orig).unsqueeze(0).to(device)
                masked_bg = bg * (1 - mask_cs)

                cond_input = torch.cat((masked_bg, keypoints), dim=1)
                gen_output = generator(cond_input, z, depth=4, alpha=1.0)
                # gen_scene = gen_output * mask_cs + masked_bg
                gen_output_resized = F.interpolate(gen_output, bg_orig.shape[-1], mode='bilinear')
                mask_resized = F.interpolate(mask_cs, bg_orig.shape[-1], mode='bilinear')
                gen_scene = gen_output_resized * mask_resized + bg_orig * (1 - mask_resized)
                gen_scene_npy = np.asarray(to_pil(gen_scene[0].to('cpu')))

                # resize generated image
                left, upper, right, lower = crop_params
                with open(fname_crop_params, 'w') as f:
                    f.write('{} {} {} {}'.format(int(left), int(upper), int(right), int(lower)))
                side = right - left
                gen_scene_npy = cv2.resize(gen_scene_npy, (side, side), interpolation=cv2.INTER_LINEAR)

                # insert person into the scene
                image_aug = np.array(to_pil(scene))
                image_aug[upper:lower, left:right] = gen_scene_npy

                # save image
                Image.fromarray(image_aug).save(fname_img)
                save_image(gen_scene, fname_crop)

            bboxes_extended.append(objects_all)
            print('{}/{} in {:.1f}s'.format(pid, len(background_paths), time.time() - start))
            if pid % 100 == 0:
                with open('extended_bboxes_1P_hard_resize_{}-{}'.format(START_SAMPLE_ID, END_SAMPLE_ID), 'wb') as f:
                    pickle.dump(bboxes_extended, f)
                print('Saved extended bounding boxes so far to "extended_bboxes_1P_hard_resize_{}-{}"'.format(
                    START_SAMPLE_ID, END_SAMPLE_ID))

    with open('extended_bboxes_1P_hard_resize_{}-{}'.format(START_SAMPLE_ID, END_SAMPLE_ID), 'wb') as f:
        pickle.dump(bboxes_extended, f)

    with open(save_file_extended_bbs, 'wb') as f:
        pickle.dump(bboxes_extended, f)
