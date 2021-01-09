import random

import cv2
import numpy as np
import torch

from dataloader import fit_to_image

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)
random.seed(0)


def xywh_intersection(xywh1, xywh2):
    '''

    :param xywh1: left, top, w, h
    :param xywh2:
    :return:
    '''
    xbox1 = [xywh1[0], xywh1[0] + xywh1[2]]
    ybox1 = [xywh1[1], xywh1[1] + xywh1[3]]
    xbox2 = [xywh2[0], xywh2[0] + xywh2[2]]
    ybox2 = [xywh2[1], xywh2[1] + xywh2[3]]

    overlaps = intersect1d(xbox1, xbox2) and intersect1d(ybox1, ybox2)
    return overlaps


def intersect1d(box1, box2):
    return box1[1] >= box2[0] and box2[1] >= box1[0]


def exp(x, a, b, c):
    res = a * np.exp(b * x) + c
    return res


def bboxes_intersect(bboxes, bbox):
    '''
    All boxes must be in XYWH format!
    :param bboxes: n bounding boxes
    :param bbox: check if this bbox intersect the others
    :return:
    '''
    for ref_bbox in bboxes:
        if xywh_intersection(ref_bbox, bbox):
            return True
    return False


def xyxy2xywh(bbs):
    xywh = []
    for xyxy in bbs:
        x1, y1, x2, y2 = xyxy
        w = x2 - x1
        h = y2 - y1
        xywh.append([x1, y1, w, h])
    xywh = np.asarray(xywh).reshape((-1, 4))
    return xywh


class PositionProposer:
    def __init__(self, gen_input_size, ground_labels=(6, 7, 8, 9),
                 height_coefs={'x_mu': 0.5, 'x_std': 0.24, 'y_mu': 0.5, 'y_std': 0.07},
                 xyxy_boxes=False, max_size=None, occlusion_classes=None, min_height=None, max_height=None,
                 insert2empty=False, ecp=False):
        self.gen_input_size = gen_input_size
        if isinstance(height_coefs, list) or isinstance(height_coefs, tuple):
            self.height_fnc = lambda y_loc: height_coefs[0] * y_loc + height_coefs[1]
        else:
            # self.height_fnc = lambda y_loc: np.exp(coefs[0] * y_loc + coefs[1])
            self.height_fnc = lambda y: -0.5682602 + 1.39825452 * y
        self.height_coefs = height_coefs
        self.min_height = 35 if min_height is None else min_height
        self.max_height = max_height
        self.allowed_labels = ground_labels  # ground, road, sidewalk, parking
        self.xyxy_boxes = xyxy_boxes
        self.max_size = max_size
        self.insert2empty = insert2empty
        self.ecp = ecp
        if occlusion_classes is not None:
            # if we have provided occlusion classes, use them
            self.allowed_labels = occlusion_classes
            self.use_occlusion = True
        else:
            self.use_occlusion = False

    def get_allowed_positions(self, segmentation, objects_all):
        ground = np.zeros_like(segmentation, dtype=bool)
        for gl in self.allowed_labels:
            tmp = segmentation == gl
            ground = np.logical_or(ground, tmp)
        ground[int(ground.shape[-2] * 0.7):, :] = False

        # ground_yx = np.where(ground)
        # ground_xy = [[x, y] for y, x in zip(ground_yx[0], ground_yx[1])]

        # mask-out areas with pedestrians
        for obj in objects_all:
            if isinstance(obj, dict):
                x, y, w, h = obj['bbox']
            else:
                x, y, w, h = obj
            x, y, w, h = int(x), int(y), int(w), int(h)
            ground[y:(y + h), x:(x + w)] = False

        return ground
        # return ground_xy, ground

    def choose_nearby(self, allowed_positions, person_bbs, mask, im_h_given=None, im_w_given=None):
        '''

        :param allowed_positions: Allowed positions where person can stand.
        :param person_bbs: annotated persons in the scene
        :param width: Width of the bounding box of the new pedestrian.
        :param mask: mask of pedestrian to insert
        :return: list of possible positions of bottom-left corner of the pedestrian bounding box
        '''
        nearby = []

        wh_y, wh_x = np.where(mask)
        sk_height = max(wh_y) - min(wh_y)

        if allowed_positions is not None:
            im_h, im_w = allowed_positions.shape
        else:
            im_h, im_w = im_h_given, im_w_given

        if allowed_positions is not None:
            bbs = [obj['bbox'] for obj in person_bbs]  # Format of one bb: [x,y,w,h] (x,y) is top-left corner.
        else:
            bbs = person_bbs['bboxes']
        if self.xyxy_boxes:
            bbs = xyxy2xywh(bbs)
        iterable = bbs

        for bb_orig in iterable:
            if allowed_positions is not None and isinstance(bb_orig, dict):
                if bb_orig['label'] != 'pedestrian':
                    continue
                bboxVis = bb_orig['bboxVis']
                bbox_orig = bb_orig['bbox']
                height = bbox_orig[-1]
                if height < 50:
                    # pedestrian is too small
                    continue
                areaVis = bboxVis[2] * bboxVis[3]
                area = bbox_orig[2] * bbox_orig[3]
                visibility = areaVis / area
                if visibility < 0.65:
                    # pedestrian is too occluded
                    continue
            else:
                bbox_orig = bb_orig.copy()

            left, right = max([bbox_orig[0], 0]), min([bbox_orig[0] + bbox_orig[2], im_w - 1])
            top, bottom = max([bbox_orig[1], 0]), min([bbox_orig[1] + bbox_orig[3], im_h - 1])
            height = bbox_orig[-1]

            # scale the skeleton so that it is the same height as the reference bounding box
            scale = height / sk_height
            tgt_size = (int(mask.shape[0] * scale), int(mask.shape[1] * scale))
            mask_scaled = cv2.resize(mask, tgt_size, interpolation=cv2.INTER_NEAREST)
            wh_y, wh_x = np.where(mask_scaled)
            if len(wh_y) == 0 or len(wh_x) == 0:
                continue
            bb_msk_h = max(wh_y) - min(wh_y)
            bb_msk_w = max(wh_x) - min(wh_x)

            # pedestrian standing on the LEFT from annotated bb
            start_x = left - bb_msk_w
            end_x = left
            if allowed_positions is not None:
                left_free = all(allowed_positions[bottom, start_x:end_x]) if end_x > 0 else False
            else:
                left_free = True

            # where would the pedestrian stand on the LEFT
            x = left - bb_msk_w - 1
            y = top
            proposed_bb = [x, y, bb_msk_w, bb_msk_h]  # TODO: compute
            right = x + bb_msk_w
            bottom = min(y + bb_msk_h, im_h - 1)

            if x < 0 or x >= im_w or y < 0 or y >= im_h or right >= im_w or bottom >= im_h:
                left_free = False

            if left_free and not bboxes_intersect(bbs, proposed_bb):
                nearby.append([start_x, bottom, height, 'left'])  # (x,y)

            # pedestrian standing on the RIGHT from annotated bb
            start_x = right
            end_x = min(right + bb_msk_w, im_w - 1)
            if allowed_positions is not None:
                right_free = all(allowed_positions[bottom, start_x:end_x]) if end_x < allowed_positions.shape[
                    1] else False
            else:
                right_free = True

            # where would the pedestrian stand on the RIGHT
            x = right + 1
            y = top
            proposed_bb = [x, y, bb_msk_w, bb_msk_h]
            right = x + bb_msk_w
            bottom = y + bb_msk_h

            if x < 0 or x >= im_w or y < 0 or y >= im_h or right >= im_w or bottom >= im_h:
                right_free = False

            if right_free and not bboxes_intersect(bbs, proposed_bb):
                nearby.append([start_x, bottom, height, 'right'])
        return nearby

    def __call__(self, image, segmentation, objects_orig, objects_all, skeleton, person_mask, return_mask=False):
        '''
        Chooses a position of a pedestrian in the image.
        :param image: PIL image of the scene
        :param person_mask: 2D numpy array of person mask
        :param segmentation: Segmentation of the scene.
        :param objects_all: Bounding boxes of the pedestrians. Format: [x,y,w,h] (x,y) is top-left corner.
        :param skeleton: Skeleton of the person that should be inserted.
        :return:
        '''
        if segmentation is not None:
            im_h, im_w = segmentation.shape[-2:]
        else:
            # retunrns image_crop, crop_size, crop_params, bbox
            return self.image_based(image, objects_all, skeleton, person_mask)

        sk_img_h, sk_img_w = skeleton.shape

        sk_y, sk_x = np.where(skeleton)
        sk_height = max(sk_y) - min(sk_y)
        sk_width = max(sk_x) - min(sk_x)

        # put pedestrian either:
        #  1) next to some already existing pedestrian => +- same height as the nearby pedestrian
        #  2) on the sidewalk or on the road => use the height fnc
        # The pedestrian must (same as in the BMVC paper):
        #  1) the pedestrian bounding box should not touch any boundary of the image
        #  2) Every point on the bottom boundary of the candidate pedestrian bounding box should belong to road/sidewalk

        ground = self.get_allowed_positions(segmentation, objects_all)  # anywhere on road/sidewalk
        if self.insert2empty:
            positions_nearby = []
        else:
            positions_nearby = self.choose_nearby(ground, objects_all, person_mask)  # next to some pedestrian
        print('Number of nearby positions: {}'.format(len(positions_nearby)))

        # change possible ground positions
        ground[:, :int(ground.shape[-1] * 0.1)] = False
        ground[:, int(ground.shape[-1] * 0.9):] = False
        ground = ground.astype(np.uint8)
        kernel = np.ones((51, 51))
        ground = cv2.erode(ground, kernel, iterations=1)

        ground_yx = np.where(ground)
        positions_ground = [[x, y] for y, x in zip(ground_yx[0], ground_yx[1])]

        use_nearby = len(positions_nearby)  # random.random() < 0.5 and
        positions = positions_nearby if use_nearby else positions_ground

        if not len(positions):
            return None

        height = -1
        while height < self.min_height:
            pos_idx = random.choice(range(len(positions)))
            if use_nearby:
                x, y, height, direction = positions[pos_idx]
            else:
                x, y = positions[pos_idx]
                direction = 'none'
                if self.ecp:
                    height = self.height_fnc(y)
                else:
                    y_tmp = y / im_h
                    height = self.height_fnc(y_tmp)
                    height = height * im_h
                if self.max_height is not None and height > self.max_height:
                    height = -1
        y_px = int((1 - y) * im_h) if isinstance(y, float) else y
        x_px = int(x * im_w) if isinstance(x, float) else x

        # how do we need to resize the inputs
        resize_factor = height / sk_height

        crop_size = int(sk_img_h * resize_factor)
        crop_shape = (crop_size, crop_size)
        crop_tmp = image[y_px - crop_size:y_px, x_px:x_px + crop_size, :]

        # resize skeleton
        skeleton_resized = cv2.resize(skeleton, crop_shape, interpolation=cv2.INTER_NEAREST)
        # find bottom left point of the skeleton that should match with (x_px, y_px)
        skr_y, skr_x = np.where(skeleton_resized)
        bottom_skr = max(skr_y)
        left_skr = min(skr_x)
        right_skr = max(skr_x)
        bottom_skr_left = max([x for x in skr_x[skr_y == bottom_skr]])
        # get the bottom left point of the image crop
        bottom_crop = y_px + (crop_size - bottom_skr)
        left_crop = x_px

        if use_nearby:
            left_crop = self.shift_left_side(direction, left_crop, left_skr, right_skr, x_px)
        else:
            left_crop = x_px - bottom_skr_left

        # crop_params in PIL crop format = (left, upper, right, lower)-tuple
        left, upper, right, lower = (left_crop, bottom_crop - crop_size, left_crop + crop_size, bottom_crop)
        left, upper, crop_size = fit_to_image(left, upper, crop_size, im_w, im_h)
        right = left + crop_size
        lower = upper + crop_size
        crop_params = left, upper, right, lower

        # use crop params to crop the image and then resize it to self.gen_input_size
        image_crop = image[upper:lower, left:right]
        image_crop_orig = image_crop.copy()
        image_crop = cv2.resize(image_crop, (self.gen_input_size, self.gen_input_size), interpolation=cv2.INTER_LINEAR)

        # resize the mask for puposes of insertion into the current segmentation
        resized_mask = cv2.resize(person_mask, (crop_size, crop_size), interpolation=cv2.INTER_NEAREST)
        augmented_segmentation, bbox = insert2segmentation(segmentation, crop_params, resized_mask)

        if return_mask:
            return image_crop, crop_size, crop_params, augmented_segmentation, bbox, image_crop_orig, resized_mask
        return image_crop, crop_size, crop_params, augmented_segmentation, bbox, image_crop_orig

    def shift_left_side(self, direction, left_crop, left_skr, right_skr, x_px):
        if direction == 'left':
            left_crop = x_px - left_skr  # x_px + right_skr // 2
        elif direction == 'right':
            left_crop = x_px - left_skr  # // 2
        elif direction == 'none':
            left_crop = x_px
        else:
            raise Exception('Something is wrong. Direction should be "left" or "right".')
        return left_crop

    def get_random_position(self, im_h, im_w):
        height = -1
        while height < 0:
            x = np.random.normal(loc=self.height_coefs['x_mu'], scale=self.height_coefs['x_std']) * im_w
            x = int(max([im_w * 0.2, min([x, im_w * 0.8])]))
            y = np.random.normal(loc=self.height_coefs['y_mu'], scale=self.height_coefs['y_std']) * im_h
            y = int(max([im_h // 10, min([y, im_h // 2])]))
            height = self.height_fnc(y / im_h) * im_h
        return x, y, height

    def image_based(self, image, objects_all, skeleton, person_mask):
        msk_y, msk_x = np.where(person_mask)
        sk_height = max(msk_y) - min(msk_y)
        sk_width = max(msk_x) - min(msk_x)

        shape = image.shape
        if shape[0] == 3:
            _, im_h, im_w = shape
        else:
            im_h, im_w, _ = shape

        # next to some pedestrian
        positions_nearby = self.choose_nearby(None, objects_all, person_mask, im_h_given=im_h, im_w_given=im_w)

        if positions_nearby:
            pos_idx = random.choice([i for i in range(len(positions_nearby))])
            x, y, height, direction = positions_nearby[pos_idx]
        else:
            x, y, height = self.get_random_position(im_h, im_w)  # TODO: generate randomly some position
            height = int(height)
            direction = 'none'
        y_px = y  # bottom
        x_px = x  # left

        resize_factor = height / sk_height

        crop_size = int(self.gen_input_size * resize_factor)
        crop_shape = (crop_size, crop_size)
        crop_tmp = image[y_px - crop_size:y_px, x_px:x_px + crop_size, :]

        # resize skeleton
        skeleton_resized = cv2.resize(skeleton, crop_shape, interpolation=cv2.INTER_NEAREST)

        # find bottom left point of the skeleton that should match with (x_px, y_px)
        skr_y, skr_x = np.where(skeleton_resized)
        bottom_skr = max(skr_y)
        left_skr = min(skr_x)
        right_skr = max(skr_x)

        # get the bottom left point of the image crop
        bottom_crop = y_px + (crop_size - bottom_skr)
        left_crop = x_px

        left_crop = self.shift_left_side(direction, left_crop, left_skr, right_skr, x_px)

        # crop_params in PIL crop format = (left, upper, right, lower)-tuple
        left, upper, right, lower = (left_crop, bottom_crop - crop_size, left_crop + crop_size, bottom_crop)
        crop_params = left, upper, right, lower
        left, upper, crop_size = fit_to_image(left, upper, crop_size, im_w, im_h)
        if self.max_size is not None:
            crop_size = min([self.max_size, crop_size])
        right = left + crop_size
        lower = upper + crop_size
        crop_params = left, upper, right, lower

        # use crop params to crop the image and then resize it to self.gen_input_size
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
            image = np.moveaxis(image, 0, 2)
        image_crop = (image)[upper:lower, left:right]
        image_crop_orig = image_crop.copy()
        image_crop = cv2.resize(image_crop, (self.gen_input_size, self.gen_input_size), interpolation=cv2.INTER_LINEAR)
        # resize the mask for puposes of insertion into the current segmentation
        resized_mask = cv2.resize(person_mask, (crop_size, crop_size), interpolation=cv2.INTER_NEAREST)
        bbox = inserted_bbox(im_h, im_w, crop_params, resized_mask, xyxy=self.xyxy_boxes)

        return image_crop, crop_size, crop_params, bbox, image_crop_orig


def insert2segmentation(segmentation, crop_params, mask, class_num=24):
    left, upper, right, lower = crop_params

    augmented = np.copy(segmentation)
    inserted_only = np.zeros_like(segmentation)

    segm_crop = augmented[upper:lower, left:right]
    tmp_crop = np.zeros_like(segm_crop)
    try:
        segm_crop[mask > 0.99] = class_num
        tmp_crop[mask > 0.99] = 1
    except:
        print('Some problem. Segm crop: {}, mask: {}, crop_params: {}'.format(segm_crop.shape, mask.shape, crop_params))

    augmented[upper:lower, left:right] = segm_crop
    inserted_only[upper:lower, left:right] = tmp_crop

    # determine the bounding box
    wh_y, wh_x = np.where(inserted_only)
    top = min(wh_y)
    left = min(wh_x)
    width = max(wh_x) - min(wh_x)
    height = max(wh_y) - min(wh_y)
    bbox = [int(left), int(top), int(width), int(height)]  # XYWH format

    return augmented, bbox


def inserted_bbox(im_h, im_w, crop_params, mask, xyxy=False):
    left, upper, right, lower = crop_params

    inserted_only = np.zeros((im_h, im_w), dtype=np.int8)

    tmp_crop = np.zeros((lower - upper, right - left), dtype=np.int8)
    try:
        tmp_crop[mask > 0.99] = 1
    except:
        print('Some problem.')

    inserted_only[upper:lower, left:right] = tmp_crop

    # determine the bounding box
    wh_y, wh_x = np.where(inserted_only)
    top = min(wh_y)
    bottom = max(wh_y)
    left = min(wh_x)
    right = max(wh_x)
    width = max(wh_x) - min(wh_x)
    height = max(wh_y) - min(wh_y)
    assert width > 2 and height > 30, "Width {}, height {}".format(width, height)
    if xyxy:
        bbox = [int(left), int(top), int(right), int(bottom)]  # XYXY format
    else:
        bbox = [int(left), int(top), int(width), int(height)]  # XYWH format

    return bbox
