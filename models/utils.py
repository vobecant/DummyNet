import collections
import random
import torch
import matplotlib.pyplot as plt

SKELETON_COCO = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], \
                 [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9], [8, 10], \
                 [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]]
SKELETON_OPENPOSE = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9],
                     [9, 10], [1, 11], [11, 12], [12, 13], [1, 0], [0, 14], [14, 16],
                     [0, 15], [15, 17], [2, 16], [5, 17]]

CONNECTIVITY = [[], []]
for conn in SKELETON_COCO:
    CONNECTIVITY[0].extend([conn[0], conn[1]])
    CONNECTIVITY[1].extend([conn[1], conn[0]])


def invert_bin_tensor(tensor):
    return torch.abs((tensor.float() - 1))


def draw_skeleton(xyv, connections=SKELETON_OPENPOSE, marker=None, get_data=False, show=False, save_name=None):
    n_skeletons = len(xyv)
    if n_skeletons > 2:
        fig, axes = plt.subplots(1, len(xyv), figsize=(100, 100))
    else:
        fig, axes = plt.subplots(1, len(xyv))
    C = ['g', 'r', 'y', 'c', 'm', 'k']

    for n, (x, y, v) in enumerate(xyv):
        v = (v > 0.5)
        if n_skeletons > 1:
            ax = axes[n]
        else:
            ax = axes
        minx, maxx = min(x[v]), max(x[v])
        cx = maxx - (maxx - minx) / 2
        miny, maxy = min(y[v]), max(y[v])
        cy = maxy - (maxy - miny) / 2
        w, h = maxx - minx, maxy - miny
        larger_side = max([w, h])
        min_side = 0.5 * larger_side
        w = max([w, min_side])
        h = max([h, min_side])
        x_from, x_to = min([minx, cx - (w / 2)]), max([maxx, cx + (w / 2)])
        y_from, y_to = min([miny, cy - (h / 2)]), max([maxy, cy + (h / 2)])

        if len(x) == 18:
            connections = SKELETON_OPENPOSE
        else:
            connections = SKELETON_COCO
        for sk_num, sk in enumerate(connections):
            if all(v[sk]):  # both keypoints are annotated
                if not all(v[sk]):  # not all keypoints are visible
                    continue
                x0, y0, x1, y1 = x[sk[0]], y[sk[0]], x[sk[1]], y[sk[1]]
                ax.plot([x0, x1], [y0, y1], '{}-'.format(random.choice(C)))
                ax.plot(x0, y0, x1, y1, 'ko' if marker is None else marker, fillstyle='full')
        ax.set_xlim(x_from, x_to)
        ax.set_ylim(y_from, y_to)
        ax.set_aspect(1.0)
    if save_name:
        plt.savefig(save_name, bbox_inches='tight')
    if show:
        plt.show()
    if get_data:
        return plt.gca().get_array()
    plt.close()


def count_trainable_parameters(model, print_info=False):
    if print_info:
        total_params = 0
        dct = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                np = p.numel()
                total_params += np
                dct[name] = np
        sorted_x = sorted(dct.items(), key=lambda kv: kv[1], reverse=True)
        sorted_x = collections.OrderedDict(sorted_x)
        for k, v in sorted_x.items():
            print('{}: {}'.format(k, v))
    else:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params
