NUM_KEYPOINTS = 17
OVERLAP = 0.3  # was 0.15
KEYPOINTS_REQUIRED = [[0, 1, 2, 3, 4],  # at least one of 'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear'
                      [5, 6],  # at least one of 'left_shoulder', 'right_shoulder'
                      [7, 8],  # at least one of 'left_elbow', 'right_elbow'
                      [11, 12],  # at least one of 'left_hip', 'right_hip'
                      [13, 14]]  # at least one of 'left_knee', 'right_knee'
KEYPOINT_NAMES = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',  # 0-4
                  'left_shoulder', 'right_shoulder',  # 5,6
                  'left_elbow', 'right_elbow',  # 7,8
                  'left_wrist', 'right_wrist',  # 9,10
                  'left_hip', 'right_hip',  # 11,12
                  'left_knee', 'right_knee',  # 13,14
                  'left_ankle', 'right_ankle']  # 15,16
KEYPOINT_NAMES_OPENPOSE = {0: 'nose', 1: 'neck', 2: 'right_shoulder', 3: 'right_elbow',
                           4: 'right_wrist', 5: 'left_shoulder', 6: 'left_elbow', 7: 'left_wrist',
                           8: 'right_hip', 9: 'right_knee', 10: 'right_ankle', 11: 'left_hip', 12: 'left_knee',
                           13: 'left_ankle', 14: 'right_eye', 15: 'left_eye', 16: 'right_ear', 17: 'left_ear'}
OPENPOSE2COCO = {0: 0, 2: 6, 3: 8, 4: 10, 5: 5, 6: 7, 7: 9, 8: 12, 9: 14, \
                 10: 16, 11: 11, 12: 13, 13: 15, 14: 2, 15: 1, 16: 4, 17: 3}
assert all([KEYPOINT_NAMES_OPENPOSE[op] == KEYPOINT_NAMES[cc] for op, cc in OPENPOSE2COCO.items()])
HEIGHT_WIDTH_RATIO = 2
CROP_SIZE = 64
NUM2SET = {0: 'val2017', 1: 'train2017', 2: 'train2014', 3: 'val2014'}
# Color code used to plot different joints and limbs (eg: joint_type=3 and
# limb_type=3 will use colors[3])
colors = [
    [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0],
    [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255],
    [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0, 255],
    [255, 0, 255], [255, 0, 170], [255, 0, 85], [255, 0, 0]]
