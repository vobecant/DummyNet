import json
import os
import random
import sys
import time

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor, ToPILImage
from torchvision.utils import save_image

from inference_utils import load_networks, encode_appearance
from loaders import Cityscapes
from models.DummyGAN import KeypointsDownsampler
from position_proposer import PositionProposer

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)
random.seed(0)

SAVE_CROPS = True
MAX_PERS_IMG = 2
START_SAMPLE_ID = 0
END_SAMPLE_ID = None
SHOW = False


class SimpleLoader(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data_file):
        self.data = torch.load(data_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


if __name__ == '__main__':

    nets_dir = sys.argv[1]  # /home/vobecant/PhD/DummyGAN_ECCV/trained_models/DummyGAN_5depth_complete_cudnn
    cs_dir = sys.argv[2]  # /home/vobecant/datasets/cityscapes
    save_dir = sys.argv[3]  # /home/vobecant/datasets/DummyGAN_cityscapes_noFine

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    to_tensor = ToTensor()
    to_pil = ToPILImage()

    # load arguments and networks
    args, generator, encoder, mask_estimator = load_networks(nets_dir)
    me_input_resizer = KeypointsDownsampler(tgt_size=64, mode='bilinear')
    generator.eval()
    generator.to(device)
    encoder.eval()
    encoder.to(device)
    mask_estimator.eval()
    mask_estimator.to(device)

    position_proposer = PositionProposer(256)

    image_size = args.image_size

    dataset = SimpleLoader(data_file='./data/YBB/test_samples_100.th')
    ybb_loader = iter(DataLoader(dataset, batch_size=1, shuffle=True))

    cityscapes_loader = Cityscapes(cs_dir, target_type='semantic')
    loader = iter(DataLoader(cityscapes_loader[1], batch_size=1, shuffle=False))

    start = time.time()

    for sample_id, sample in enumerate(cityscapes_loader[0], 1):
        if START_SAMPLE_ID and sample_id < START_SAMPLE_ID:
            continue
        if END_SAMPLE_ID and sample_id == END_SAMPLE_ID:
            break
        image_path, image, segmentation = sample
        ext_len = len('_leftImg8bit.jpg')
        if '_leftImg8bit.png' != image_path[-ext_len:] and '_leftImg8bit.jpg' != image_path[ext_len:]:
            print('Skip {}. Invalid extension {}'.format(image_path, image_path[-ext_len:]))
            continue
        dirs, image_name = os.path.split(image_path)
        city = dirs.split('/')[-1]

        print('\nimage {}/{} {}'.format(sample_id, len(cityscapes_loader), image_path))

        image = np.asarray(image)
        segmentation = np.asarray(segmentation)
        bb_path = image_path.replace('_leftImg8bit.png', '_gtBboxCityPersons.json').replace('/leftImg8bit/',
                                                                                            '/gtBboxCityPersons/')
        with open(bb_path, 'r') as f:
            ann = json.load(f)
            objects_orig = ann['objects']
            objects_all = objects_orig.copy()
        for person_num in range(1, MAX_PERS_IMG + 1):
            print('\tAdd person {}'.format(person_num))

            # set paths
            images_dir = os.path.join(save_dir, '{}P'.format(person_num), 'images')
            labels_dir = os.path.join(save_dir, '{}P'.format(person_num), 'labels')
            bboxes_dir = os.path.join(save_dir, '{}P'.format(person_num), 'bboxes')
            crops_dir = os.path.join(save_dir, '{}P'.format(person_num), 'crops')
            images_dir_city = os.path.join(images_dir, city)
            if not os.path.exists(images_dir_city):
                os.makedirs(images_dir_city)
            labels_dir_city = os.path.join(labels_dir, city)
            if not os.path.exists(labels_dir_city):
                os.makedirs(labels_dir_city)
            bboxes_dir_city = os.path.join(bboxes_dir, city)
            if not os.path.exists(bboxes_dir_city):
                os.makedirs(bboxes_dir_city)
            crops_dir_city = os.path.join(crops_dir, city)
            if not os.path.exists(crops_dir_city):
                os.makedirs(crops_dir_city)

            fname_img = os.path.join(images_dir_city, image_name)
            fname_crop = os.path.join(crops_dir_city, image_name.replace('.png', '_{}P.png'.format(person_num)))
            fname_crop_params = fname_crop.replace('.png', '.txt')
            fname_segm = os.path.join(labels_dir_city, image_name.replace('_leftImg8bit.png', '_gtFine_labelIds.png'))
            fname_bbox = os.path.join(bboxes_dir_city, os.path.split(bb_path)[1])

            try:
                conditioned_sample = next(loader)
            except StopIteration:
                loader = iter(DataLoader(dataset, batch_size=1, shuffle=False))
                conditioned_sample = next(loader)
            keypoints = conditioned_sample['keypoints'].to(device)
            skeleton = torch.sum(keypoints, dim=(0, 1)).clamp_(0, 1).cpu().numpy()
            person = conditioned_sample['image'].to(device)
            mask = conditioned_sample['mask'].to(device)
            sk_y, sk_x = np.where(skeleton)

            with torch.no_grad():
                # mask = estimate_mask(mask_estimator, me_input_resizer, keypoints)
                # mask = F.interpolate(mask, keypoints.shape[-2:], mode='bilinear')
                # skeleton = torch.sum(keypoints, dim=(0, 1)).clamp_(0, 1).cpu().numpy()
                proposal = position_proposer(image, segmentation, objects_orig, objects_all, skeleton,
                                             mask.squeeze().cpu().numpy())
                if proposal is None:
                    print('Didnt find any possible placement. Do not augment.')
                    cv2.imwrite(fname_segm, segmentation)
                    Image.fromarray(image).save(fname_img)
                    continue

                image_crop, crop_size, crop_params, augmented_segmentation, bbox, image_crop_orig = proposal

                # add new object to the existing ones
                object_ann = {'instance_id': None, 'bbox': bbox, 'bboxVis': bbox, 'label': 'pedestrian'}
                objects_all.append(object_ann)
                cur_ann = {'imgHeight': ann['imgHeight'], 'imgWidth': ann['imgWidth'], 'objects': objects_all}
                with open(fname_bbox, 'w') as f:
                    json.dump(cur_ann, f)

                # save segmentation
                cv2.imwrite(fname_segm, augmented_segmentation)

                # get conditional appearance
                masked_person = person * mask
                masked_person_npy = np.asarray(to_pil(masked_person[0].cpu()))
                z = encode_appearance(encoder, masked_person, 64)

                bg = to_tensor(image_crop).unsqueeze(0).to(device)
                masked_bg = bg * (1 - mask)
                cond_input = torch.cat((masked_bg, keypoints), dim=1)
                gen_output = generator(cond_input, z, depth=4, alpha=1.0)
                # mask = mask**3
                gen_scene = gen_output * mask + bg * (1 - mask)
                gen_scene_npy = np.asarray(to_pil(gen_scene[0].to('cpu')))

            # resize generated image
            left, upper, right, lower = crop_params
            with open(fname_crop_params, 'w') as f:
                f.write('{} {} {} {}'.format(int(left), int(upper), int(right), int(lower)))
            side = right - left
            gen_scene_npy = cv2.resize(gen_scene_npy, (side, side), interpolation=cv2.INTER_LINEAR)

            # insert person into the scene
            image_aug = np.copy(image)
            image_aug[upper:lower, left:right] = gen_scene_npy

            if SHOW:
                import matplotlib

                matplotlib.use('tkagg')
                import matplotlib.pyplot as plt

                fig = plt.figure(figsize=(15, 4))
                fig.add_subplot(1, 5, 1)
                plt.imshow(image_crop)
                fig.add_subplot(1, 5, 2)
                plt.imshow(skeleton, 'gray')
                fig.add_subplot(1, 5, 3)
                plt.imshow(mask.squeeze().numpy(), 'gray')
                fig.add_subplot(1, 5, 4)
                plt.imshow(masked_person_npy)
                fig.add_subplot(1, 5, 5)
                plt.imshow(gen_scene_npy)

                plt.figure()
                plt.imshow(image_aug)
                plt.show()
                plt.close('all')

            # save image
            Image.fromarray(image_aug).save(fname_img)
            save_image(gen_scene, fname_crop)

            # to be used for next added persons
            image = image_aug
            segmentation = augmented_segmentation
        print('{}/{} in {:.1f}s'.format(sample_id, len(cityscapes_loader), time.time() - start))
