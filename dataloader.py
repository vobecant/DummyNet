import argparse
import os
import sys
from random import randint

import matplotlib
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as tfunc
from PIL import Image

#matplotlib.use('tkagg')
import cv2

from torchvision.transforms import ToTensor

import json
import base64
import time

import math

from models.constants import *

DEBUG, DEBUG_KPTS, DEBUG_MORPHOLOGY = False, False, False
DEBUG_CROPS = False
SHOW_STATS = False


def valid_annotation(person_skeleton, visibility, required_skeleton_height, required_joints_coco, min_num_visible):
    sk_x, sk_y = [x for x, y in person_skeleton], [y for x, y in person_skeleton]
    sk_w = max(sk_x) - min(sk_x)
    sk_h = max(sk_y) - min(sk_y)
    sk_size = max([sk_w, sk_h])
    if sk_size < required_skeleton_height:
        # print('Too little: skeleton max size: {}<{}'.format(sk_size, required_skeleton_height))
        return False

    visible_joints = [joint_idx for joint_idx, v in enumerate(visibility) if v]

    if min_num_visible and len(visible_joints) < min_num_visible:
        # too few visible joints
        return False

    if len(required_joints_coco) > 0:
        for group in required_joints_coco:
            intersection = set(visible_joints).intersection(group)
            if len(intersection) == 0:
                # print('No enough kpts. Visible: {}, required: {}.'.format(visible_joints, required_joints_coco))
                return False

    return True


class Cityscapes_crops(data.Dataset):
    def __init__(self, samples_dir, ids=None, min_visible=-1):
        '''

        :param samples_dir:
        :param ids: (%min, %max) of ids to use
        '''
        import pickle
        files = os.listdir(samples_dir)
        self.samples = [os.path.join(samples_dir, f) for f in files if 'pkl' in f]
        if ids is not None:
            n_samples = len(self.samples)
            min_id = int(ids[0] * n_samples)
            max_id = int(ids[1] * n_samples)
            self.samples = self.samples[min_id:max_id]
            print('Keep indices {}-{}'.format(min_id, max_id))
        if min_visible > 0:
            print('Keep only samples with at least {} annotated joints!'.format(min_visible))
            filtered = []
            for i, sn in enumerate(self.samples, 1):
                with open(sn, 'rb') as f:
                    sample = pickle.load(f)
                keypoints_npy = sample['mask_keypoint']
                visible = self.count_joints(keypoints_npy)
                if visible >= min_visible:
                    filtered.append(sn)
                if i % 100 == 0:
                    print(i)

            print('Kept {}/{}'.format(len(filtered), len(self.samples)))
            self.samples = filtered
        self.to_tensor = ToTensor()

    def count_joints(self, keypoints_map):
        count = 0
        for kpt_map in keypoints_map:
            if np.any(kpt_map):
                count += 1
        return count

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        path = self.samples[item]
        with open(path, 'rb') as f:
            sample = pickle.load(f)

        image_npy = sample['image']
        orig_mask_npy = sample['mask_orig']
        keypoints_npy = sample['mask_keypoint']

        assert image_npy.max() > 1.0, image_npy.max()
        image_t = self.to_tensor(image_npy).float()
        orig_mask_t = torch.from_numpy(orig_mask_npy).clamp_(0, 1).unsqueeze(0).float()
        keypoints_t = torch.from_numpy(keypoints_npy).clamp_(0, 1).float()

        assert image_t.max() <= 1.0, image_t.max()
        assert orig_mask_t.max() <= 1.0, orig_mask_t.max()
        assert keypoints_t.max() <= 1.0, keypoints_t.max()

        data = {'image': image_t, 'mask_orig': orig_mask_t, 'mask_keypoint': keypoints_t, 'path': path}
        return data


class YBB_Dataset(data.Dataset):
    def __init__(self, images_folder, transformations, crop_size, overlap, scale_range, rot_max, flip_prob, min_sk_size,
                 required_joints_coco, split_json=None, min_conf=None, max_blur=None, min_darkness=None,
                 max_darkness=None, precomputed_data_path=None, debug=False, load_from='image', images_dir=None,
                 annot_subfolder_name=None, idx_range=None, img_ext='.jpg', mask_ext='.pbm', masks_dir=None,
                 min_num_visible=None):

        self.required_joints_coco = required_joints_coco
        self.base_folder = images_folder
        self.min_sk_size = min_sk_size
        self.transformations = transformations
        self.crop_size = crop_size
        self.overlap = overlap
        self.min_conf = min_conf  # minimal confidence of the detected skeleton
        self.max_blur = max_blur
        self.min_darkness = min_darkness
        self.max_darkness = max_darkness
        self.debug = debug  # if debug is True, show images of samples
        self.idx_range = idx_range
        self.min_num_visible = min_num_visible
        self.transformer = Transformer(scale_range, rot_max, flip_prob)
        assert load_from in ['video', 'image']
        print('YBB dataset loads from {}'.format(load_from))
        self.load_from = load_from
        self.images_dir = images_dir

        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.masks_dir = masks_dir

        # keep it here
        self.joint_to_limb_heatmap_relationship = [
            [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10],
            [1, 11], [11, 12], [12, 13], [1, 0], [0, 14], [14, 16], [0, 15], [15, 17],
            [2, 16], [5, 17]]

        start_t = time.time()
        last_t = start_t

        if precomputed_data_path and os.path.isfile(precomputed_data_path):
            self.load_YBB(precomputed_data_path, split_json)
        else:
            self.precompute_YBB(annot_subfolder_name, idx_range, images_folder, last_t, min_sk_size,
                                precomputed_data_path, required_joints_coco, split_json, start_t)

    def split_trn_val(self, val_fraction=0.05):
        '''
        Splits idx to TRN indices for training and to VAL indices for validation
        :param val_fraction:
        :return:
        '''
        np.random.seed(42)
        shuffled = np.random.shuffle(self.idx)
        n_trn = int(len(self.idx) / (1 - val_fraction))
        self.trn_idx = list(shuffled[:n_trn])
        self.val_idx = list(shuffled[n_trn:])

    def load_YBB(self, precomputed_data_path, split_json):
        print('Loading precomputed data from {}'.format(precomputed_data_path))
        start = time.time()
        with open(precomputed_data_path) as json_file:
            loaded_json = json.load(json_file)
        self.idx = loaded_json
        with open(split_json, 'r') as infile:
            self.data = json.load(infile)
        self.counter_valid = len(self.idx)
        print('Files loaded in {:.1f}s! Data size: {}'.format(time.time() - start, self.counter_valid))

    def precompute_YBB(self, annot_subfolder_name, idx_range, images_folder, last_t, min_sk_size, precomputed_data_path,
                       required_joints_coco, split_json, start_t):
        print('Precomputed file does not exists, create a new one.')
        folders = [folder for folder in os.listdir(images_folder) if '_img' in folder]
        if split_json is not None:
            print('Parse given JSON split {}'.format(split_json))
            self.parse_split_json(split_json)
        else:
            # find all json files
            print('Find all JSON files...')
            self.json_files = []
            for i, folder in enumerate(folders):
                if idx_range is not None:
                    if i < idx_range[0] or idx_range[1] > i:
                        continue
                folder_path = os.path.join(images_folder, folder)
                if annot_subfolder_name is not None:
                    # subfolder that contains JSON files with annotations
                    folder_path = os.path.join(folder_path, annot_subfolder_name)
                files = os.listdir(folder_path)
                json_files_folder = [os.path.join(folder_path, f) for f in files if f[-5:] == '.json']
                self.json_files.extend(json_files_folder)

            # create indexes
            self.idx = []
            self.counter_valid = 0
            counter_all = 0
            print('Start to parse {} JSON files with annotations.'.format(len(self.json_files)))
            max_dark_over = 0
            for file_no, fname in enumerate(self.json_files):
                with open(fname, 'r') as infile:
                    loaded_json = json.load(infile)
                    json_frames = loaded_json["frames"]
                    if 'resolution' not in loaded_json.keys():
                        print('\tFile {} is corrupted (no "resolution"). Skip it.'.format(fname), file=sys.stderr)
                        continue
                    im_h, im_w = loaded_json['resolution'][:2]
                    try:
                        n_person_list = [self.get_n_person(fr['person_to_joint_assoc']) for fr in json_frames]
                    except:
                        print('\tFile {} is corrupted (wrong coding?). Skip it.'.format(fname), file=sys.stderr)
                        continue
                for annotated_frame_num, frame in enumerate(json_frames):
                    n_person = self.get_n_person(frame['person_to_joint_assoc'])
                    joint_list = self.Base64Decode(frame["joint_list"])
                    person_to_joint_assoc = self.Base64Decode(frame["person_to_joint_assoc"])
                    blur_amount = frame['blur_amount']
                    darkness = frame['darkness']
                    for person_idx in range(n_person):
                        counter_all += 1
                        _, person_joints_img, visibility, conf = get_person_joints(person_idx, joint_list,
                                                                                   person_to_joint_assoc, im_h,
                                                                                   im_w)
                        if (self.min_conf is not None) and (conf < self.min_conf):
                            # too low detection confidence
                            print('Too low confidency.')
                            continue
                        if (self.max_blur is not None) and (blur_amount > self.max_blur):
                            # too blurry image
                            print('Too blury.')
                            continue
                        if (self.min_darkness is not None) and (darkness < self.min_darkness):
                            # too dark
                            print('Too dark.')
                            continue
                        if (self.max_darkness is not None) and (darkness > self.max_darkness):
                            # too bright
                            max_dark_over += 1
                            continue
                        visibility_coco = openpose2coco_order(visibility)
                        if not valid_annotation(person_joints_img, visibility_coco, min_sk_size,
                                                required_joints_coco, self.min_num_visible):
                            # not enough of detected keypoints or too small
                            print('Not enough annotated keypoints in file {}, frame {}, person {}'.format(file_no,
                                                                                                          annotated_frame_num,
                                                                                                          person_idx))
                            continue
                        record = (file_no, annotated_frame_num, person_idx)
                        self.idx.append(record)
                        self.counter_valid += 1
                if (file_no + 1) % 100 == 0:
                    now = time.time()
                    total = now - start_t
                    last_100 = now - last_t
                    last_per_file = last_100 / 100
                    print('\t{}/{}\ttotal: {:.1f} s, last 100: {:.1f} s, per file: {:.2f} s'.format
                          (file_no + 1, len(self.json_files), total, last_100, last_per_file))
                    last_t = now
            elapsed = time.time() - start_t
            percentage = self.counter_valid / counter_all * 100
            print('{} skipped because they were too bright.'.format(max_dark_over))
            print('Initialization of BGG Dataset took {}.\n\tAnnotated skeletons:\t{}'
                  '\n\tValid skeletons:\t{} ({:.2f}%)'.format(elapsed, counter_all, self.counter_valid, percentage))

        # save precomputed data
        json_dump = json.dumps(self.idx)
        directory = os.path.dirname(precomputed_data_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(precomputed_data_path, "w") as f:
            f.write(json_dump)
        print('Saved precomputed data to {}'.format(precomputed_data_path))

    def parse_split_json(self, split_json):
        self.idx = []
        with open(split_json, 'r') as infile:
            self.data = json.load(infile)
        print('Start parsing given split. Number of frames: {}'.format(len(self.data['frames'])))
        st = time.time()
        st_last100 = time.time()
        skipped_bright = 0
        for data_idx, frame in enumerate(self.data['frames']):
            person_id = frame['person_id']
            darkness = frame['darkness']
            person_to_joint_assoc = self.Base64Decode(frame['person_to_joint_assoc'])
            im_h, im_w, im_c = frame['resolution']  #
            joint_list = self.Base64Decode(frame['joint_list'])

            _, person_joints_img, visibility, conf = get_person_joints(person_id, joint_list,
                                                                       person_to_joint_assoc, im_h,
                                                                       im_w)

            if (self.min_conf is not None) and (conf < self.min_conf):
                # too low detection confidence
                print('Too low conf.')
                continue
            if (self.max_darkness is not None) and (darkness > self.max_darkness):
                # print('darkness={} (>{})'.format(darkness, self.max_darkness))
                skipped_bright += 1
                continue
            visibility_coco = openpose2coco_order(visibility)
            is_valid_ann = valid_annotation(person_joints_img, visibility_coco, self.min_sk_size,
                                            self.required_joints_coco, self.min_num_visible)
            if not is_valid_ann:
                # not enough of detected keypoints or too small
                continue

            # all the checks passed, save the index to self.data
            self.idx.append(data_idx)

            # print
            if (data_idx + 1) % 100 == 0:
                now = time.time()
                total = now - st
                last = now - st_last100
                print(
                    '{}/{}. Total time: {:.1f}s, last 100: {:.2f}s'.format(data_idx + 1, len(self.data['frames']),
                                                                           total, last))
                st_last100 = time.time()

        elapsed = time.time() - st
        print('Skipped {} because they were too bright.'.format(skipped_bright))
        print('Filtered annotations in {:.2}s. Kept {}/{}.'.format(elapsed, len(self.idx), len(self.data['frames'])))

    def segment_data(self, mask_rcnn, save_path, idx_given=None, mask_suffix='.pbm'):
        st = time.time()
        st_last = st
        info = []
        skipped = 0
        n_files = 0
        n_files_total = len(idx_given) if idx_given is not None else len(self.idx)
        indices = idx_given if idx_given else range(len(self.idx))
        print('Start to segment {} images.'.format(len(indices)))
        for idx in indices:
            # print('.', end='.', flush=True)
            n_files += 1
            batch = self.__getitem__(idx)
            fname = batch['dataset_name']
            path, img_fname = os.path.split(fname)
            vid_name = path.split(os.sep)[-1]
            frame_num = batch['frame_num']
            keypoints = batch['mask_keypoint']
            thin_mask = batch['mask_estim_thin']
            person_id = batch['person_id']

            mask_fname = img_fname.replace(self.img_ext, '_{}{}'.format(person_id, self.mask_ext))
            directory = os.path.join(save_path, vid_name)
            if not os.path.exists(directory):
                os.makedirs(directory)
            fname_mask = os.path.join(directory, mask_fname)
            if os.path.exists(fname_mask):
                print('{} already exists. Continue...')
                continue

            # get crop parameters
            crop_params = batch['crop_params']
            x1, y1, side = crop_params
            x2, y2 = x1 + side, y1 + side
            crop_params = (x1, y1, x2, y2)

            # get frame
            frame, im_h, im_w = get_frame(fname, frame_num, load_from='image')
            frame = np.asarray(frame)

            _, masks, masks_cropped, conf, masked_images, cropped_images = \
                mask_rcnn.run_only_selected(frame, save=False, min_height=self.min_sk_size, return_masks_in_image=True)

            if masks is None:
                if n_files % 100 == 0:
                    total = time.time() - st
                    last = time.time() - st_last
                    estim_left = (n_files / total * len(indices) - total) / 3600
                    print(
                        '\n{}/{}. Didn\'t detect person in {}. last 100 in {:.1f}s, estimated time left {:.2f}h'.format(
                            n_files, n_files_total, skipped, last, estim_left))
                    st_last = time.time()
                # print('Unable to segment person in image no. {}'.format(idx))
                skipped += 1
                continue

            # TODO: find the segmentation that belongs to this annotation
            #       it is the one with the most keypoints inside
            mask_img = self.choose_pedestrian_segmentation(keypoints, masks, crop_params)
            if mask_img is None:
                if n_files % 100 == 0:
                    total = time.time() - st
                    last = time.time() - st_last
                    estim_left = (n_files / total * len(indices) - total) / 3600
                    print(
                        '\n{}/{}. Didn\'t detect person in {}. last 100 in {:.1f}s, estimated time left {:.2f}h'.format(
                            n_files, n_files_total, skipped, last, estim_left))
                    st_last = time.time()
                # print('the detected mask doesnt overlay well the detected keypoints in image no. {}'.format(idx))
                skipped += 1
                continue

            mask_cropped_rcnn = cv2.resize(mask_img[y1:y2, x1:x2], (self.crop_size, self.crop_size),
                                           interpolation=cv2.INTER_NEAREST)
            mask_estim_kpts = batch['mask_orig'][0]
            # joint_mask = np.logical_or(mask_cropped_rcnn, mask_estim_kpts).numpy()
            joint_mask = np.logical_and(mask_cropped_rcnn, mask_estim_kpts).numpy()
            # joint_mask = np.logical_or(joint_mask, thin_mask).numpy()

            # TODO: save the segmentation
            cv2.imwrite(fname_mask, joint_mask)
            # plt.imsave(fname, mask_img)
            data_idx = self.idx[idx]

            frame_num_from_idx = self.data['frames'][data_idx]['frame_num']
            person_id = self.data['frames'][data_idx]['person_id']
            assert frame_num_from_idx == frame_num
            idx_info = {'idx': idx, 'fname': fname, 'frame_num': frame_num, 'person_id': person_id}
            info.append(idx_info)

            if n_files % 100 == 0:
                total = time.time() - st
                last = time.time() - st_last
                estim_left = (n_files / total * len(indices) - total) / 3600
                print('\n{}/{}. Didn\'t detect person in {}. last 100 in {:.1f}s, estimated time left {:.2f}h'.format(
                    n_files, n_files_total, skipped, last, estim_left))
                st_last = time.time()

        info_fname = os.path.join(save_path, 'info.json')
        with open(info_fname, 'w') as fout:
            json.dump(info, fout)
        elapsed = time.time() - st
        print(
            'Processed {} files in {:.1f}s. Segmentation unsuccessful for {} images.'.format(n_files, elapsed, skipped))

    def choose_pedestrian_segmentation(self, keypoints, masks, crop_params, fraction_required=0.6):
        x1, y1, x2, y2 = crop_params
        choosen_pedstrian = -1
        max_kpts_inside = 0
        for i, mask in enumerate(masks):
            cropped_mask = mask[y1:y2, x1:x2]
            inside, annotated_kpts = self.kpts_inside(keypoints, cropped_mask)
            if int(annotated_kpts * 0.6) > inside:
                # too few of annotated keypoints inside detected mask
                continue
            if inside > max_kpts_inside:
                choosen_pedstrian = i
                max_kpts_inside = inside
        if choosen_pedstrian == -1:
            # the detected mask doesnt overlay well the detected keypoints
            return None
        else:
            return masks[choosen_pedstrian]

    def kpts_inside(self, keypoints, mask):
        resized_mask = cv2.resize(mask, keypoints.shape[1:], interpolation=cv2.INTER_NEAREST)
        inside = 0
        annotated_kpts = 0
        for kpt_map in keypoints:
            if (kpt_map > 0).any():
                annotated_kpts += 1
            if np.logical_and(kpt_map, resized_mask).any():
                inside += 1
        return inside, annotated_kpts

    def get_n_person(self, p2j_assoc):
        tmp = p2j_assoc.split('",')[-1][2:-2]
        strnum = tmp.split(',')[0]
        try:
            num = int(strnum)
        except:
            print(p2j_assoc)
        return num

    def Base64Encode(self, ndarray):
        return json.dumps([str(ndarray.dtype), base64.b64encode(ndarray).decode('utf-8'), ndarray.shape])

    # for loading ndarray from json file
    def Base64Decode(self, jsonDump):
        loaded = json.loads(jsonDump)
        dtype = np.dtype(loaded[0])
        arr = np.frombuffer(base64.decodebytes(bytearray(loaded[1], 'utf-8')), dtype)
        if len(loaded) > 2:
            return arr.reshape(loaded[2])
        return arr

    def parse_json(self, json_file):
        with open(json_file, 'r') as infile:
            loaded_json = json.load(infile)
        for frame_num in range(loaded_json["frames"]):
            frame = loaded_json["frames"][frame_num]

    def parse_frame(self, json_file, annotated_frame_num):
        '''

        :param json_file:
        :param annotated_frame_num: Ordinal number of a frame among the annotated ones. Does not need to be equal to the frame number (not all the frames are annotated).
        :return:
        '''
        with open(json_file, 'r') as infile:
            loaded_json = json.load(infile)
        frame = loaded_json["frames"][annotated_frame_num]
        joint_list_decoded = self.Base64Decode(frame["joint_list"])
        person_to_joint_decoded = self.Base64Decode(frame["person_to_joint_assoc"])
        frame_num_loaded, joint_list, person_to_joint_assoc = frame["frame_num"], \
                                                              joint_list_decoded, person_to_joint_decoded
        return frame_num_loaded, joint_list, person_to_joint_assoc

    def __len__(self):
        return len(self.idx)
        if self.train:
            return len(self.trn_idx)
        else:
            return len(self.val_idx)

    def json2name(self, json_name, frame_num_loaded, load_from, ending=".mp4", to_rem='pose_and_bb.json'):
        fname = json_name.replace(to_rem, '')
        path, fname = os.path.split(fname)
        if load_from == 'video':
            vid_name = fname + ending
            return vid_name
        else:
            path = os.path.join(self.images_dir, path, fname)
            img_fname = '{}.jpg'.format(format(frame_num_loaded, '05'))
            if 'filtered' in path:
                path = path.replace('filtered', 'img')
            img_path = os.path.join(path, img_fname)
            return img_path

    def show_detections(self, img_orig, joint_list, person_joint_info, bool_fast_plot=True, plot_ear_to_shoulder=False,
                        idx=0):
        NUM_LIMBS = len(self.joint_to_limb_heatmap_relationship)
        img_orig_np = img_orig.copy() if isinstance(img_orig, np.ndarray) else np.asarray(img_orig)
        canvas = img_orig_np.copy()
        to_plot = cv2.addWeighted(img_orig_np, 0.3, canvas, 0.7, 0)

        limb_thickness = 4
        # Last 2 limbs connect ears with shoulders and this looks very weird.
        # Disabled by default to be consistent with original rtpose output
        which_limbs_to_plot = NUM_LIMBS if plot_ear_to_shoulder else NUM_LIMBS - 2
        for limb_type in range(which_limbs_to_plot):
            joint_indices = person_joint_info[self.joint_to_limb_heatmap_relationship[limb_type]].astype(
                int)

            # joint_coords[:,0] represents Y coords of both joints;
            # joint_coords[:,1], X coords
            joint_coords = joint_list[joint_indices, 0:2]

            for joint_id, joint in zip(joint_indices, joint_coords):  # Draw circles at every joint
                if joint_id == -1:
                    continue
                cv2.circle(canvas, tuple(joint[0:2].astype(
                    int)), 3, (255, 255, 255), thickness=-1)

            if -1 in joint_indices:
                # Only draw actual limbs (connected joints), skip if not
                # connected
                continue

            # mean along the axis=0 computes meanYcoord and meanXcoord -> Round
            # and make int to avoid errors
            coords_center = tuple(
                np.round(np.mean(joint_coords, 0)).astype(int))
            # joint_coords[0,:] is the coords of joint_src; joint_coords[1,:]
            # is the coords of joint_dst
            limb_dir = joint_coords[0, :] - joint_coords[1, :]
            limb_length = np.linalg.norm(limb_dir)
            # Get the angle of limb_dir in degrees using atan2(limb_dir_x,
            # limb_dir_y)
            angle = math.degrees(math.atan2(limb_dir[1], limb_dir[0]))

            # For faster plotting, just plot over canvas instead of constantly
            # copying it
            cur_canvas = canvas if bool_fast_plot else canvas.copy()
            polygon = cv2.ellipse2Poly(
                coords_center, (int(limb_length / 2), int(round(limb_thickness / 4))), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, colors[limb_type])
            if not bool_fast_plot:
                canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

        matplotlib.use('tkagg')
        plt.imsave('/home/vobecant/checkanns/{}.png'.format(idx), canvas)

    def mask_pedestrian(self, image, mask, mask_thr, to_tensor=False, return_mask_estim=False):
        '''
        mask_bin = np.ones_like(mask)
        mask_bin[mask < mask_thr] = 0
        image_masked = -np.ones_like(image)
        image_masked[mask_bin == 0] = np.array(image)[mask_bin == 0]
        if to_tensor:
            image_masked = Image.fromarray(image_masked)
            image_masked = tfunc.to_tensor(image_masked)
        mask_bin = torch.from_numpy(mask_bin)
        return image_masked, mask_bin
        '''
        if len(mask.shape) == 2:
            mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        image_masked = np.ones_like(image) * 255 * mask + image * (1 - mask)
        mask_bin = np.ones_like(mask)
        mask_bin[mask < mask_thr] = 0
        mask_bin = torch.from_numpy(mask_bin[:, :, 0])
        mask_estim = torch.from_numpy(mask[:, :, 0].copy())
        if to_tensor:
            if image_masked.dtype == np.float64:
                image_masked = image_masked / image_masked.max()
                image_masked = (image_masked * 255).astype(np.uint8)
            image_masked = Image.fromarray(image_masked)
            image_masked = tfunc.to_tensor(image_masked)
        if return_mask_estim:
            return image_masked, mask_estim
        else:
            return image_masked, mask_bin

    def get_crop_parameters(self, joints, valid, img_w, img_h, overlap):
        '''
        params:
            joints: list of joints in image coordinates, each joint is (x,y) numpy array
            valid: boolean list, indicates whether a joint is valid
            img_w, img_h: image width and height
            overlap: how much should the crop of pedestrian be bigger than the pedestrian itself
        '''
        valid_joints = joints[valid]
        xs, ys = [x for x, y in valid_joints], [y for x, y in valid_joints]

        # get min and max values
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        # get width and height
        width = x_max - x_min
        height = y_max - y_min

        # we want to crop square box => make both width and height equal to the larger of them
        square_side = max([width, height])

        # multiply by overlap
        square_side = int(square_side * overlap)
        half_side = square_side // 2

        # get center of the image
        x_c = x_min + width // 2
        y_c = y_min + height // 2

        # ensure that the square is completely inside the image
        x, y = x_c - half_side, y_c - half_side
        x, y, square_side = fit_to_image(x, y, square_side, img_w, img_h)

        return x, y, square_side

    def random_idx(self):
        return randint(0, len(self.idx) - 1)

    def get_data(self, index, type='new'):
        start_t = time.time()
        if type == 'new':
            assert index < len(self.idx)
            data_idx = self.idx[index]
            frame = self.data['frames'][data_idx]

            fname = frame['video_name']
            person_id = frame['person_id']
            frame_num = frame['frame_num']
            person_to_joint_assoc = self.Base64Decode(frame['person_to_joint_assoc'])
            joint_list = self.Base64Decode(frame['joint_list'])

            return fname, person_id, frame_num, joint_list, person_to_joint_assoc, start_t
        else:
            file_num, annotated_frame_num, person_id = self.idx[index]
            fname = self.json_files[file_num]
            if self.debug:
                print('fname: {}, f_num: {}, fr_num: {}, person_id: {}'.format(fname, file_num, annotated_frame_num,
                                                                               person_id))

            frame_num_loaded, joint_list, person_to_joint_assoc = self.parse_frame(fname, annotated_frame_num)

            return fname, person_id, frame_num_loaded, joint_list, person_to_joint_assoc, start_t

    def __getitem__(self, index, mask_estim_thr=0.5, load_type='new', train=True):
        # try:
        fname, person_id, frame_num, joint_list, person_to_joint_assoc, start_t = self.get_data(index, type=load_type)

        fname = os.path.join(self.images_dir, format(os.sep).join(fname.split(os.sep)[-3:]))
        fname = self.json2name(fname, frame_num, self.load_from)

        # load frame
        fts = time.time()
        frame, img_h, img_w = get_frame(fname, frame_num, self.load_from)
        frame_load_t = time.time() - fts

        # transform to image coordinates
        mult = np.array([img_w, img_h])
        joint_list_img = (joint_list[:, :2] * mult).astype(int)

        # get person joints
        person_joints_rel, person_joints_img, valid, conf = get_person_joints(person_id, joint_list,
                                                                              person_to_joint_assoc, img_h,
                                                                              img_w)

        if self.debug:
            # plt.imshow(frame)
            # plt.show()
            # self.show_detections(frame, joint_list_img, person_to_joint_assoc[person_id][:18])
            pass

        # TODO: get crop parameters (x,y,side)
        crop_parameters = self.get_crop_parameters(person_joints_img, valid, img_w, img_h, self.overlap)

        # TODO: cut out just the image with pedestrian
        cropped = crop_image(frame, crop_parameters)

        # TODO: shift joints such that they correspond to the crop
        joints_shifted = shift_joints(person_joints_img, crop_parameters)
        joints_shifted_all = shift_joints(joint_list_img, crop_parameters)

        # TODO: transform image to PIL image
        if not isinstance(cropped, Image.Image):
            try:
                image = Image.fromarray(cropped.astype('uint8'), 'RGB')
            except:
                raise Exception('"Cropped" shape: {}, crop parameters: {}, "frame" shape: {}'.format(cropped.shape,
                                                                                                     crop_parameters,
                                                                                                     frame.shape))
        else:
            image = cropped

        # TODO: resize
        orig_img_size = image.size[0]
        image = image.resize((self.crop_size, self.crop_size), resample=Image.BILINEAR)
        joints_shifted_resized = resize_joints(joints_shifted, orig_img_size, self.crop_size)
        joints_shifted_all_resized = resize_joints(joints_shifted_all, orig_img_size, self.crop_size)

        if self.debug:
            # self.show_detections(cropped, joints_shifted_all, person_to_joint_assoc[person_id][:18])
            self.show_detections(image, joints_shifted_all_resized, person_to_joint_assoc[person_id][:18], idx=index)

        # TODO: estimate mask
        # first check if more precise mask is stored

        directory, mask_file = os.path.split(fname)
        vid_name = directory.split(os.sep)[-1]
        mask_file = mask_file.replace(self.img_ext, '_{}{}'.format(person_id, self.mask_ext))
        mask_file_path = os.path.join(self.masks_dir, vid_name, mask_file)
        if os.path.isfile(mask_file_path):
            if self.debug:
                print('Use mask from Mask RCNN')
            mask_estim = cv2.imread(mask_file_path, cv2.IMREAD_GRAYSCALE)
            mask_estim = cv2.resize(mask_estim, (self.crop_size, self.crop_size))
        else:
            mask_estim = get_skeleton_mask_openpose(joints_shifted_all_resized, joints_shifted_resized,
                                                    person_to_joint_assoc[person_id][:18], valid, self.crop_size,
                                                    show=False)
            # save the mask
            # cv2.imwrite(mask_file_path, mask_estim)

        # TODO: mask augmentation
        # mask_estim = self.augment_mask(mask_estim)

        mask_estim_thin = np.ones_like(mask_estim)
        mask_estim_thin[mask_estim < 0.7] = 0
        mask_estim_thin = torch.from_numpy(mask_estim_thin.astype(np.uint8))

        # TODO: mask the pedestrian
        masked, binary_mask = self.mask_pedestrian(image, mask_estim, mask_estim_thr)

        if self.debug:
            # plt.imshow(masked)
            # plt.show()
            pass

        # TODO: create layers with keypoint locations
        joints_coco_order = openpose2coco_order(joints_shifted_resized)
        valid_coco_order = openpose2coco_order(valid)
        keypoint_channels = get_keypoint_masks(joints_coco_order, (self.crop_size, self.crop_size),
                                               joints=joints_coco_order, visibility=valid_coco_order)
        keypoint_channels = torch.Tensor(keypoint_channels)

        if self.debug:
            # keypoint_img = torch.sum(keypoint_channels, dim=0)
            # show_img_t(keypoint_img)
            pass

        # TODO: transform everything
        self.transformer.reset_params()  # IMPORTANT!!! So that we have the same parameters for all the transformations!
        img_t = self.transformer.transform(image)
        mask_t = self.transformer.transform(binary_mask).unsqueeze(0).float()
        image_masked_t, mask_estim = self.mask_pedestrian(image, mask_estim, mask_estim_thr, to_tensor=True,
                                                          return_mask_estim=True)
        mask_blurred_t = self.transformer.transform(mask_estim).unsqueeze(0).float()
        image_masked_t = self.transformer.transform(image_masked_t)
        keypoints_t = self.transformer.transform_keypoints(keypoint_channels)
        mask_estim_thin = self.transformer.transform(mask_estim_thin)

        if self.debug:
            # show_img_t(img_t)
            # show_img_t(mask_t)
            # keypoint_img_trans = torch.sum(keypoints_t, dim=0)
            # show_img_t(keypoint_img_trans)
            pass

        data = {'image': img_t, 'image_masked': image_masked_t, 'mask_orig': mask_t, 'mask_keypoint': keypoints_t,
                'dataset_name': fname, 'frame_num': frame_num, 'crop_params': crop_parameters,
                'mask_estim_thin': mask_estim_thin, 'person_id': person_id, 'mask_fname': mask_file_path,
                'mask_blurred': mask_blurred_t}

        elapsed = time.time() - start_t
        return data

    def augment_mask(self, mask):

        mask_size = mask.shape[1]
        min_k_size = max([mask_size // 30, 1])
        max_k_size = max([mask_size // 15, 1])

        k_size = randint(min_k_size, max_k_size)
        sigma = k_size // 1.5
        if k_size > 0:
            if k_size % 2 == 0:
                k_size += 1
            kernel = (k_size, k_size)
            k_size_openning = max([1, k_size // 2])
            kernel_openning = (k_size_openning, k_size_openning)
            opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_openning)

            dilated = cv2.dilate(opened, kernel, iterations=1)
            blurred = cv2.GaussianBlur(dilated, kernel, sigma)
            mask = blurred

        mask = mask / mask.max()
        return mask

    def export_paths(self, export_file, base_crop_dir, load_type='new'):
        directory, fname = os.path.split(export_file)
        if not os.path.exists(directory):
            os.makedirs(directory)

        n_images = self.__len__()
        paths = []
        with open(export_file, 'w') as f:
            for index in range(n_images):
                fname, person_id, frame_num, joint_list, person_to_joint_assoc, start_t = self.get_data(index,
                                                                                                        type=load_type)
                fname = os.path.join(self.images_dir, format(os.sep).join(fname.split(os.sep)[-3:]))
                fname = self.json2name(fname, frame_num, self.load_from)
                directory, img_name = os.path.split(fname)
                video_name = directory.split(os.sep)[-1]
                img_name, ext = img_name.split('.')
                img_name = '{}_{}.{}'.format(img_name, person_id, ext)
                fname = os.path.join(base_crop_dir, video_name, img_name)

                paths.append(fname)
                if (index + 1) != self.__len__():
                    fname += '\n'
                f.write(fname)
        np.save(export_file.replace('txt', 'npy'), paths)


class Transformer:
    def __init__(self, scale_range, max_rotation, flip_prob):
        self.flip_prob = flip_prob
        self.scale_range = scale_range
        self.max_rotation = max_rotation
        self.scale = None
        self.angle = None
        self.to_tensor = transforms.ToTensor()

    def scale_sampler(self):
        return random_from_range(self.scale_range[0], self.scale_range[1])

    def angle_sampler(self):
        return random_from_range(-self.max_rotation, self.max_rotation)

    def flip_sampler(self):
        return True if random_from_range(0, 1) < self.flip_prob else False

    def affine_transformer(self, t, angle, scale):
        return tfunc.affine(t, angle=angle, translate=(0, 0), scale=scale, shear=0, resample=Image.BILINEAR)

    def random_flipper(self, t):
        return tfunc.hflip(t)

    def reset_params(self):
        self.angle = self.angle_sampler()
        self.scale = self.scale_sampler()
        self.flip = self.flip_sampler()

    def transform(self, x):
        if self.max_rotation == 0 and self.scale_range[0] == 1 and self.scale_range[1] == 1 and self.flip_prob == 0:
            if type(x) != torch.Tensor:
                if isinstance(x, np.ndarray):
                    x = torch.from_numpy(x).float()
                else:
                    x = self.to_tensor(x)
            return x
        remove_first_dim = False
        if type(x) == torch.Tensor:
            if x.shape[0] != 3:
                if len(x.shape) != 2:
                    raise Exception('Unexpected shape!')
                x = x.unsqueeze(0).float()
                remove_first_dim = True
            x = tfunc.to_pil_image(x)
        elif type(x) == np.ndarray:
            x = Image.fromarray(x)
            remove_first_dim = True

        # perform transformations
        after_affine = self.affine_transformer(x, self.angle, self.scale)
        after_flip = self.random_flipper(after_affine) if self.flip else after_affine

        tensor = self.to_tensor(after_flip)

        if remove_first_dim:
            tensor = tensor[0]

        return tensor

    def transform_keypoints(self, keypoint_channels):
        transformed_list = []
        for keypoint_loc in keypoint_channels:
            transformed = self.transform(keypoint_loc)
            transformed_list.append(transformed)
        keypoint_channels = torch.stack(transformed_list)
        return keypoint_channels


def random_from_range(min_val, max_val):
    rnd_num = torch.FloatTensor(1).uniform_(min_val, max_val)
    return rnd_num


def resize_joints(joints, im_size, crop_size):
    resized = []
    divider = im_size / crop_size
    for joint in joints:
        resized_joint = (joint / divider).astype(int)
        resized.append(resized_joint)
    resized = np.asarray(resized)
    return resized


def get_person_joints(person_id, joint_list, person_to_joint_assoc, height, width):
    person_joint_info = person_to_joint_assoc[person_id]
    joint_indices = person_joint_info[:18].astype(int)

    # joint coordinates in [0,1]
    joint_coords_rel = joint_list[joint_indices, 0:2]

    # joint coordinates in image
    multiplier = np.array([width, height])
    joint_coords_img = (joint_coords_rel * multiplier).astype(int)

    # valid joints
    valid = joint_indices != -1

    # confidence
    confidence = person_joint_info[-2]

    return joint_coords_rel, joint_coords_img, valid, confidence


class get_reader_imageio:
    def __init__(self, video_name):
        self.video_name = video_name

    def __enter__(self):
        self.reader = imageio.get_reader(self.video_name)
        return self.reader

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.reader.close()


def get_frame(file_name, frame_no, load_from, debug=False):
    '''
    params:
        file_name: img_path to the file (image or video)
        frame_no: number of frame that should be returned
    '''
    if load_from == 'video':
        start_t = time.time()
        try:
            # imageio backend
            st = time.time()
            with get_reader_imageio(file_name) as reader:
                wait = time.time() - st
                frame = reader.get_data(frame_no)
                img_h, img_w, _ = frame.shape
            # OpenCV backend
            '''
            capture = cv2.VideoCapture(video_name)
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            _, frame = capture.read()
            capture.realease()
            '''
        except:
            raise Exception('***Exception*** caught during loading frame {} from video {}.'.format(frame_no, file_name))
        elapsed = time.time() - start_t
        if debug:
            print('Elapsed for loading a frame: {:.3f}, wait time: {:.3f}'.format(elapsed, wait))

    else:
        # file_name = vidpath2imgpath(file_name, frame_no)
        frame = Image.open(file_name)
        img_w, img_h = frame.size

    return frame, img_h, img_w


def openpose2coco_order(in_op_order):
    coco_order = [0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10]
    in_coco_order = [in_op_order[i] for i in coco_order]
    return in_coco_order


def shift_joints(joints, parameters):
    x, y, _ = parameters
    shift = np.array([x, y])
    shifted_joints = joints - shift
    return shifted_joints


def crop_image(image, parameters):
    x1, y1, side = parameters
    x2, y2 = x1 + side, y1 + side
    if isinstance(image, Image.Image):
        # box – a 4-tuple defining the left, upper, right, and lower pixel coordinate.
        box = (x1, y1, x2, y2)
        cropped = image.crop(box)
    else:
        # order in numpy: y, x, channels
        cropped = image[y1:y2, x1:x2, :]
    return cropped


def fit_to_image(x, y, side, im_w, im_h):
    '''
    params:
        (x,y): top-left corner of the crop
        side: crop size of a square box in px
        im_w, im_h: image width and height
    output:
        modified values (x,y) such that the whole box is inside an image
    '''

    # move the top-left corner so that it is valid first
    if x < 0:
        x = 0
    if y < 0:
        y = 0

    # coordinates of the bottom right corned
    x2, y2 = x + side, y + side
    xsize = abs(x - x2)
    ysize = abs(y - y2)

    max_x, max_y = im_w - 1, im_h - 1

    if x2 > max_x:
        # delta: how much do we need to shift the top-left x-coordinate
        delta_x = x2 - max_x
        x = max([0, x - delta_x])
        x2 = min([max_x, x2])
        xsize = abs(x - x2)
    if y2 > max_y:
        delta_y = y2 - max_y
        y = max([0, y - delta_y])
        y2 = min([max_y, y2])
        ysize = abs(y - y2)

    if (x < 0) or (y < 0):
        raise Exception(
            "x and y cannot be negative! (x={:d}, x2={}, dx={}, y={:d}, y2={:d}, dy={}; "
            "image: {}x{})".format(x, x2, delta_x, y, y2, delta_y, im_h, im_w))

    orig_side = side
    side = min([xsize, ysize])
    if orig_side != side:
        pass
        # print('Original size {}, current size: {}.'.format(orig_side, side), file=sys.stderr)

    # return modified values of the top-left corner
    return x, y, side


def show_image(image, color=None, title=None):
    plt.figure()
    if color is not None:
        plt.imshow(image, color)
    else:
        plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.show()


def get_keypoint_masks(kp, im_size, size_frac=40, joints=None, visibility=None, head_ids=[0, 1, 2, 3, 4]):
    '''
    Creates masks of keypoints. One mask per keypoint.
    :param kp: array of keypoint information
    :param im_size: size of image
    :return: array containing masks of keypoints
    '''
    x = kp[0::3] if joints is None else [x for x, y in joints]
    y = kp[1::3] if joints is None else [y for x, y in joints]
    visibility = kp[2::3] if visibility is None else [2 if v else 0 for v in visibility]
    size = int(np.ceil(im_size[0] / size_frac))  # int(get_widths(y, visibility)[-1] / 2)  # [largest, medium, smallest]
    size = max([size, 1])
    size_head = int(size * 0.75)
    size_head = max([size_head, 1])
    masks = []
    for id, v in enumerate(visibility):
        mask = np.zeros(im_size)
        mask.fill(0)
        if v > 0:
            center = (int(x[id]), int(y[id]))
            value = 1  # 0.5 if v == 1 else 1
            if head_ids is not None and id in head_ids:
                cv2.circle(mask, center, size_head, value, -1)
            else:
                cv2.circle(mask, center, size, value, -1)
        masks.append(mask)
    return masks


def get_skeleton_mask_openpose(joints_openpose_all, joints_openpose_person, person_joint_info, visibility, im_size,
                               plot_ear_to_shoulder=False, show=False):
    joint_to_limb_heatmap_relationship = [
        [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10],
        [1, 11], [11, 12], [12, 13], [1, 0], [0, 14], [14, 16], [0, 15], [15, 17],
        [2, 16], [5, 17]]

    xs, ys, vs = [x for x, y in joints_openpose_person], [y for x, y in joints_openpose_person], [2 if v else 0 for v in
                                                                                                  visibility]

    # predefine masks
    mask_estim = np.zeros((im_size, im_size))
    mask_estim.fill(0)
    # draw line
    widths = get_widths(ys, vs)  # [largest, medium, smallest]
    values = [0.25, 0.5, 1.0]
    head_multipliers = [1.5, 1.25, 1]
    nose_visible = vs[0]
    shoulders_visible = np.all([vs[2], vs[5]])
    # TODO: find the connections
    not_show = None  # do not show connections between keypoint on the head
    # parameters of head
    center, axis, sh_hip_dist = get_ellipse_params(xs, ys, vs, head_idx=[0, 14, 15, 16, 17], shoulder_ids=[2, 5],
                                                   knee_ids=[9, 12], hip_ids=[8, 11], ret_shoulder_hip_dist=True)

    if nose_visible and shoulders_visible:
        # add connection from nose to a point between shoulders
        pt_x = int(xs[5] + (xs[6] - xs[5]) / 2)
        pt_y = int(ys[5] + (ys[6] - ys[5]) / 2)
        rotation = np.rad2deg(np.arctan2(pt_y - center[1], pt_x - center[0]))
    else:
        if shoulders_visible:
            pt_x = int(xs[5] + (xs[6] - xs[5]) / 2)
            pt_y = int(ys[5] + (ys[6] - ys[5]) / 2)
        else:
            visible_shoulder_id = 5 if vs[5] == 2 else 6
            pt_x = xs[visible_shoulder_id]
            pt_y = ys[visible_shoulder_id]
        rotation = None if center is None else np.rad2deg(np.arctan2(pt_y - center[1], pt_x - center[0]))

    num_limbs = len(joint_to_limb_heatmap_relationship)
    which_limbs_to_plot = num_limbs if plot_ear_to_shoulder else num_limbs - 2

    # draw circles in the positions where the feet should be

    # draw limbs and circles in the positions where the feet should be
    feet_idx = [10, 13]
    radiuses = [int(m * sh_hip_dist) for m in [0.5, 0.4, 0.3]]
    head_multipliers, widths, values, radiuses = [head_multipliers[-1]], [widths[-1]], [values[-1]], [radiuses[-1]]
    for mul, width, val, ft_rad in zip(head_multipliers, widths, values, radiuses):
        assert width > 0, "Width is <= 0!"
        for feet in feet_idx:
            if vs[feet] < 1:
                continue
            x, y = joints_openpose_person[feet, 0], joints_openpose_person[feet, 1]
            cv2.circle(mask_estim, (x, y), radius=ft_rad, color=val, thickness=-1)
        for limb_type in range(which_limbs_to_plot):
            joint_indices = person_joint_info[joint_to_limb_heatmap_relationship[limb_type]].astype(int)
            if -1 in joint_indices:
                # Only draw actual limbs (connected joints), skip if not
                # connected
                continue
            # joint_coords[:,0] represents Y coords of both joints;
            # joint_coords[:,1], X coords
            joint_coords = joints_openpose_all[joint_indices, 0:2]
            x, y = joint_coords[:, 0], joint_coords[:, 1]
            from_pt, to_pt = (x[0], y[0]), (x[1], y[1])
            cv2.line(mask_estim, from_pt, to_pt, val, width)
        if center is not None:
            center = (center[0], center[1])
            tmp_ax = (int(mul * axis[0]), int(mul * axis[1]))
            cv2.ellipse(mask_estim, center, tmp_ax, rotation, 0, 360, val, -1)

    # draw torso
    body_idx = [5, 2, 8, 11]
    poly_pts = get_torso_points(xs, ys, vs, body_idx=body_idx)
    cv2.fillConvexPoly(mask_estim, poly_pts, 1)

    # draw head
    if show:
        plt.imshow(mask_estim)
        plt.show()

    return mask_estim


def get_ellipse_params(x, y, v, head_idx=[0, 1, 2, 3, 4], shoulder_ids=[5, 6], knee_ids=[11, 12], hip_ids=[13, 14],
                       ret_shoulder_hip_dist=False):
    '''
    Computes parameters of an ellipse that is used for head.
    :param x: keypoints' x-coordinates
    :param y: keypoints' y-coordinates
    :param v: keypoints' visibility information
    :param head_idx: ('nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear')
    :return: tuple (x,y) of center, tuple (major, minor) of parameters of ellipse axes
    '''
    annotated = [v[i] > 0 for i in head_idx]
    visible_pos = None if not any(annotated) else np.asarray(
        [[[x[idx], y[idx]]] for idx, vis in zip(head_idx, annotated) if vis])

    # get used shoulder point
    if v[shoulder_ids[0]] == 2 and v[shoulder_ids[1]] == 2:
        shoulder_x = np.mean([x[shoulder_ids[0]], x[shoulder_ids[1]]])
        shoulder_y = np.mean([y[shoulder_ids[0]], y[shoulder_ids[1]]])
    else:
        used_id_shoulder = shoulder_ids[0] if v[shoulder_ids[0]] == 2 else shoulder_ids[1]
        shoulder_x = x[used_id_shoulder]
        shoulder_y = y[used_id_shoulder]

    # get knee point
    if v[knee_ids[0]] == 2 and v[knee_ids[1]] == 2:
        knee_x = np.mean([x[knee_ids[0]], x[knee_ids[1]]])
        knee_y = np.mean([y[knee_ids[0]], y[knee_ids[1]]])
    else:
        used_id_knee = knee_ids[0] if v[knee_ids[0]] == 2 else knee_ids[1]
        knee_x = x[used_id_knee]
        knee_y = y[used_id_knee]

    # get hip point
    if v[hip_ids[0]] == 2 and v[hip_ids[1]] == 2:
        hip_x = np.mean([x[hip_ids[0]], x[hip_ids[1]]])
        hip_y = np.mean([y[hip_ids[0]], y[hip_ids[1]]])
    else:
        used_id_hip = hip_ids[0] if v[hip_ids[0]] == 2 else hip_ids[1]
        hip_x = x[used_id_hip]
        hip_y = y[used_id_hip]

    major_shoulder_hip = int(abs(shoulder_y - knee_y) / 4)
    major_knee_hip = int(0.4 * abs(hip_y - knee_y))
    major = np.mean([major_shoulder_hip, major_knee_hip])
    minor = 0.75 * major
    center = np.mean(visible_pos, axis=0, dtype=int)[0] if any(annotated) else None
    center = (int(center[0]), int(center[1])) if center is not None else None

    if ret_shoulder_hip_dist:
        shoulder_hip_dist = (shoulder_x - hip_x) ** 2 + (shoulder_y - hip_y) ** 2
        shoulder_hip_dist = np.sqrt(shoulder_hip_dist)
        return center, (int(major), int(minor)), shoulder_hip_dist
    else:
        return center, (int(major), int(minor))


def get_torso_points(xs, ys, vis, body_idx=[5, 6, 12, 11]):
    '''
    Determines annotated torso points.
    :param xs: keypoints' x-coordinates
    :param ys: keypoints' y-coordinates
    :param vis: keypoints' visibility information
    :return:
    '''
    body_pts = []
    for idx in body_idx:
        if vis[idx] > 0:  # torso part is annotated
            body_pts.append([xs[idx], ys[idx]])
    return np.asarray(body_pts)


def get_widths(ys, vis):
    '''
    Computes widths of lines for drawing pedestrian masks based on keypoints.
    :param ys: y-values of keypoints
    :param vis: visibility information about each keypoint
    :return:
    '''
    # we are not interested in ankles since we do not require them to be visisble
    ys, vis = ys[:15], vis[:15]
    ys = [y for (y, v) in zip(ys, vis) if v > 0]  # chooses y-coords of only visible points
    y_min, y_max = np.min(ys), np.max(ys)
    height = y_max - y_min  # in pixels
    base = int(height / 10)
    widths = [max(4 * base, 1), max(int(2.5 * base), 1), max(base, 1)]
    return widths


def load_maskrcnn(img_suffix=None, mask_suffix=None, save_folder=None, model_res=None):
    from maskrcnn.person_detector.person_detector import COCODemoPerson as COCODemo
    from maskrcnn.maskrcnn_benchmark.config import cfg

    res = 500 if model_res is None else model_res

    parser = argparse.ArgumentParser(description="PyTorch Object Detection Webcam Demo")
    parser.add_argument(
        "--config-file",
        default="./maskrcnn/configs/caffe2/MY_e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml",
        metavar="FILE",
        help="img_path to config file",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.7,
        help="Minimum score for the prediction to be shown",
    )
    parser.add_argument(
        "--min-image-size",
        type=int,
        default=res,
        help="Smallest size of the image to feed to the model. "
             "Model was trained with 800, which gives best results",
    )
    parser.add_argument(
        "--show-mask-heatmaps",
        dest="show_mask_heatmaps",
        help="Show a heatmap probability for the top masks-per-dim masks",
        action="store_true",
    )
    parser.add_argument(
        "--masks-per-dim",
        type=int,
        default=2,
        help="Number of heatmaps per dimension to show",
    )
    parser.add_argument(
        "opts",
        help="Modify model config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--min_height",
        type=int,
        default=128,
        help="Minimum height of segmented pedestrian.",
    )

    args = parser.parse_args()
    print('Run Mask RCNN with resolution {}'.format(args.min_image_size))

    # load config from file and command-line arguments
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # prepare object that handles inference plus adds predictions on top of image
    mask_rcnn = COCODemo(
        cfg,
        img_suffix,
        mask_suffix,
        save_folder,
        confidence_threshold=args.confidence_threshold,
        show_mask_heatmaps=args.show_mask_heatmaps,
        masks_per_dim=args.masks_per_dim,
        min_image_size=args.min_image_size,
    )

    return mask_rcnn
