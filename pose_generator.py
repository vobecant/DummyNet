# Author: David Hurych, david.hurych@gmail.com
#         February 2, 2021

import json
import joblib
import matplotlib.pyplot as plt
import numpy as np

# Heatmap indices to find each limb (joint connection). Eg: limb_type=1 is
# Neck->LShoulder, so joint_to_limb_heatmap_relationship[1] represents the
# indices of heatmaps to look for joints: neck=1, LShoulder=5
joint_to_limb_heatmap_relationship = [
    [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10],
    [1, 11], [11, 12], [12, 13], [1, 0], [0, 14], [14, 16], [0, 15], [15, 17],
    [2, 16], [5, 17]]

NUM_JOINTS = 18
NUM_LIMBS = len(joint_to_limb_heatmap_relationship)
EPS = 0.0000000001


# vizualization
def plot_pose(joint_list, active_joints, view_clust, pose_num, id, fig, opt):
    plot_ear_to_shoulder = False
    limb_thickness = 4

    # create person_to_joint_assoc
    person_to_joint_assoc = np.zeros(18, dtype=int)
    counter = 0
    for k in range(18):  # 18 skeleton joints
        if (active_joints[k] == True):
            person_to_joint_assoc[k] = counter
        else:
            person_to_joint_assoc[k] = -1
        counter += 1

    plt.clf()
    ax = plt.subplot(111)

    ax.set_aspect(1.0)
    ax.set_xlim(np.min(joint_list[:, 0]) - 5, np.max(joint_list[:, 0]) + 5)
    # Last 2 limbs connect ears with shoulders and this looks very weird.
    # Disabled by default to be consistent with original rtpose output
    which_limbs_to_plot = NUM_LIMBS if plot_ear_to_shoulder else NUM_LIMBS - 2
    for limb_type in range(which_limbs_to_plot):
        person_joint_info = person_to_joint_assoc[:]
        joint_indices = person_joint_info[joint_to_limb_heatmap_relationship[limb_type]].astype(int)

        if -1 in joint_indices:
            # Only draw actual limbs (connected joints), skip if not connected
            continue

        joint_coords = joint_list[joint_indices, 0:2]

        if limb_type == 0:
            color = 'cyan'
        elif limb_type == 1:
            color = 'blue'
        elif limb_type == 2:
            color = 'lime'
        elif limb_type == 3:
            color = 'green'
        elif limb_type == 4:
            color = 'chocolate'
        elif limb_type == 5:
            color = 'brown'
        elif limb_type == 6:
            color = 'purple'
        elif limb_type == 7:
            color = 'salmon'
        elif limb_type == 8:
            color = 'lightblue'
        elif limb_type == 9:
            color = 'darkblue'
        elif limb_type == 10:
            color = 'goldenrod'
        elif limb_type == 11:
            color = 'navy'
        elif limb_type == 12:
            color = 'olive'
        elif limb_type == 13:
            color = 'orange'
        elif limb_type == 14:
            color = 'orangered'
        elif limb_type == 15:
            color = 'teal'
        elif limb_type == 16:
            color = 'tan'
        elif limb_type == 17:
            color = 'wheat'
        elif limb_type == 18:
            color = 'yellowgreen'

        ax.plot(joint_coords[:, 0], -joint_coords[:, 1], color, linewidth=4)
        for joint in joint_coords:  # Draw circles at every joint
            ax.plot(joint[0], -joint[1], 'ro', markersize=7)

    plt.axis('off')

    if opt['save_img']:
        fig.savefig(opt['save_path'] + 'v%3.3i' % view_clust + '_p%5.5i' % pose_num + '_id%4.4i.jpg' % id,
                    bbox_inches='tight')

    plt.show(block=False)
    plt.pause(0.2)


# relative pose [0, 1][0, 1] to absolute pose [0 n_rows][0 n_cols]
def get_abs_pose(joint_list, opt):
    n_cols = opt['n_cols']
    n_rows = opt['n_rows']

    center_x = round((n_cols) / 2) - 1
    center_y = round((n_rows) / 2) - 1

    # get height equal to 1
    joint_list_max = joint_list.max(0)
    joint_list_min = joint_list.min(0)

    scale1 = joint_list_max[1] - joint_list_min[1]
    scale2 = (n_rows - 1) * opt['scale_height']  # margins
    joint_list[:, 0] -= ((joint_list_max[0] + joint_list_min[0]) / 2)
    joint_list[:, 1] -= ((joint_list_max[1] + joint_list_min[1]) / 2)
    joint_list = np.round((joint_list) * (scale2 / (scale1 + EPS)))

    # update max, min
    joint_list_max = joint_list.max(0)
    joint_list_min = joint_list.min(0)

    # scale if left or right points are still out of image width
    if joint_list_max[0] > n_cols / 2 or joint_list_min[0] < -n_cols / 2 + 1:
        scale3 = (n_cols - 1) / (
                    (n_cols - 1) + 2 * max([joint_list_max[0] - (n_cols / 2), abs(joint_list_min[0] - 1 + n_cols / 2)]))
        joint_list = np.round(joint_list * scale3)

    joint_list[:, 0] += center_x
    joint_list[:, 1] += center_y

    assert (joint_list.max(0)[0] <= n_cols - 1 and joint_list.min(0)[0] >= 0), 'joint list x out of bounds'
    assert (joint_list.max(0)[1] <= n_rows - 1 and joint_list.min(0)[1] >= 0), 'joint list y out of bounds'

    return joint_list.astype(np.int32)


# get list of visible joints
def get_active_joints(active_joints_list, num_of_samples, opt):
    assert (opt['visibility_sampling_strategy'] == 'distribution' or opt[
        'visibility_sampling_strategy'] == 'max'), 'ERROR: unknown visibility_sampling_strategy'

    if opt['visibility_sampling_strategy'] == 'distribution':
        # sample randomly in indexes of training samples visibility
        idx = np.round(np.random.rand(num_of_samples) * (len(active_joints_list) - 1)).astype(int)

    if opt['visibility_sampling_strategy'] == 'max':
        # get visibility vector with max number of joints visible
        idx = (np.ones([num_of_samples, 1], dtype=int) * np.argmax(np.sum(np.array(active_joints_list), 1))).ravel()

    final_list = []
    for i in range(num_of_samples):
        final_list.append(active_joints_list[idx[i]])

    return final_list


# calculates number of samples to generate per cluster given strategy
def get_distribution(joints_pca, opt):
    tr_samples_histogram = np.zeros([len(joints_pca['clusters'])], dtype=int)
    samples_distribution = np.zeros([len(joints_pca['clusters'])], dtype=int)
    num_empty = 0

    for i in range((len(joints_pca['clusters']))):
        if (opt['cluster_sampling_strategy'] == 'distribution'):  # sample from original pose distribution
            tr_samples_histogram[i] = joints_pca['clusters'][i]['num_samples']
        elif (opt['cluster_sampling_strategy'] == 'uniform'):  # sample uniformly from pose clusters
            tr_samples_histogram[i] = (1 if joints_pca['clusters'][i]['num_samples'] > 0 else 0)
        else:
            print('ERROR: unknown cluster sampling strategy')

    cumulative = np.cumsum(tr_samples_histogram)
    probability = cumulative / cumulative[-1]

    # x = np.sort(np.random.uniform(0, 1, opt['num_of_samples']))
    x = np.linspace(0, 1, opt['num_of_samples'], endpoint=True)
    last_idx = 0
    for i in range(len(x)):
        for j in range(last_idx, len(probability), 1):
            last_idx = j
            if (x[i] <= probability[j]):
                samples_distribution[j] += 1
                break

    print('Samples generated = ' + str(np.sum(samples_distribution)))

    # Uncomment to vizualize distributions
    # plt.figure()
    # plt.plot(tr_samples_histogram/cumulative[-1])
    # plt.plot(samples_distribution/opt['num_of_samples'])
    # plt.show()

    return samples_distribution


# generates poses from training data distribution
def generate_skeletons(joints_pca, opt):
    data = {}
    data['clusters'] = []
    data['larger_than_20'] = []  # generated samples (PCA)
    data['between_0_20'] = []  # direct samples
    data['equal_to_0'] = []  # empty clusters

    for i in range(len(joints_pca['clusters'])):
        data['clusters'].append({'visibility': [], 'poses': []})

    if opt['vizu'] == True:
        fig = plt.figure()

    # get the number of samples to generate per each cluster
    samples_per_cluster = get_distribution(joints_pca, opt)

    for i in range(len(joints_pca['clusters'])):

        if i % 100 == 0:
            print('Empty clusters = ' + str(len(data['equal_to_0'])))
            print('Clusters with direct samples = ' + str(len(data['between_0_20'])))
            print('Clusters with PCA generated samples = ' + str(len(data['larger_than_20'])))

        joints_mat = joints_pca['clusters'][i]['joints_mat']

        # empty clusters appear when some poses were present only in videos that were used in testing set of our Youtube BB split
        if len(joints_mat) == 0:
            data['equal_to_0'].append(i)
            continue

        print('------------------ new cluster ' + str(i) + ' ------------------')
        # small clusters with less than 21 examples are sampled directly from data points
        num_of_samples = joints_mat.shape[0]
        if num_of_samples < 21:  # for less than 21 samples, the PCA was not calculated
            data['between_0_20'].append(i)

            # get random from training skeletons
            rand_idx = (np.round(np.random.rand(samples_per_cluster[i]) * (num_of_samples - 1))).astype(int)
            for j in range(samples_per_cluster[i]):
                print('view cluster: ' + str(joints_pca['clusters'][i]['cluster_view']) + ', pose cluster: ' + str(
                    i) + ', sample number: ' + str(rand_idx[j]) + ', direct sample')

                joint_list = np.zeros((18, 2), dtype=float)
                joint_list[:, 0] = joints_mat[rand_idx[j].astype(int), 0:18].astype(float)
                joint_list[:, 1] = joints_mat[rand_idx[j].astype(int), 18:36].astype(float)
                joint_list = get_abs_pose(joint_list, opt)

                if opt['vizu'] == True:
                    plot_pose(joint_list, joints_pca['clusters'][i]['active_joints'][rand_idx[j]][:],
                              joints_pca['clusters'][i]['cluster_view'], i, j, fig, opt)

                data['clusters'][i]['poses'].append([joint_list[:, 0].tolist(), joint_list[:, 1].tolist()])
                data['clusters'][i]['visibility'].append(
                    (joints_pca['clusters'][i]['active_joints'][rand_idx[j]].astype(int)).tolist())

            continue

        data['larger_than_20'].append(i)

        # load scaler and pca
        scaler = joblib.load(opt['load_path'] + 'pca_per_cluster/sclaer_obj_%6.6i' % i)
        pca = joblib.load(opt['load_path'] + 'pca_per_cluster/pca_obj_%6.6i' % i)

        params_max = joints_pca['clusters'][i]['params_max']
        params_min = joints_pca['clusters'][i]['params_min']
        params_diff = params_max - params_min

        active_joints_sampled = get_active_joints(joints_pca['clusters'][i]['active_joints'], samples_per_cluster[i], opt)

        for j in range(samples_per_cluster[i]):

            print('view cluster: ' + str(joints_pca['clusters'][i]['cluster_view']) + ', pose cluster: ' + str(
                i) + ', sample number: ' + str(j))
            if (samples_per_cluster[i] == 1):
                current = params_min + params_diff * 0.5
            else:
                current = params_min + params_diff * (j / (samples_per_cluster[i] - 1))

            p = scaler.inverse_transform(pca.inverse_transform(current)[:])
            joint_list = np.zeros((18, 2), dtype=float)
            joint_list[:, 0] = p[0:18].astype(float)
            joint_list[:, 1] = p[18:36].astype(float)
            joint_list = get_abs_pose(joint_list, opt)

            if opt['vizu'] == True:
                plot_pose(joint_list, active_joints_sampled[j][:], joints_pca['clusters'][i]['cluster_view'], i, j, fig, opt)

            data['clusters'][i]['poses'].append([joint_list[:, 0].tolist(), joint_list[:, 1].tolist()])
            data['clusters'][i]['visibility'].append(active_joints_sampled[j].tolist())

    print('Empty clusters = ' + str(len(data['equal_to_0'])))
    print('Clusters with direct samples = ' + str(len(data['between_0_20'])))
    print('Clusters with PCA generated samples = ' + str(len(data['larger_than_20'])))
    return data


def main():
    opt = {}
    opt['save'] = True  # save generated list of skeletons?
    opt['save_suffix'] = ''
    opt['vizu'] = True  # vizualize generated samples?
    opt['n_rows'] = 256  # height
    opt['n_cols'] = 256  # width
    opt['scale_height'] = 0.8  # def margins of skeleton from top, down edges wrt n_rows
    opt['save_img'] = False  # save skeleton vizualization? If True, then opt['vizu'] also has to be True to take effect
    opt['load_path'] = 'D:/data/dummynet/clustering_results_unique_vid_tst_exp2_2019-11-12/'  # path to
    opt['save_path'] = 'D:/data/dummynet/clustering_results_unique_vid_tst_exp2_2019-11-12/tmp/'  # where to save results
    opt['cluster_sampling_strategy'] = 'distribution'  
    # distribution - each cluster generates a portion of the n samples corresponding
    #                to portion of training samples used to create this cluster
    # uniform - gives more diverse poses. Each cluster generates the same number of samples (except for empty clusters),
    #           smoothly sample from PCA calculated parameters from min to max and take
    #           proportion of samples from clusters where PCA is not calculated (from less than 21 training samples)
    opt['visibility_sampling_strategy'] = 'distribution'
    # max - takes always the maximum amount of visible joints - valid for cluster with PCA calculated,
    #       for direct samples it takes always corresponding visibility vector
    # distribution - samples from the real training samples visibility vectors distribution
    opt['num_of_samples'] = 10000  # actual number of skeletons to generate from 25193 (non-empty) pose clusters

    np.random.seed(42)

    # load data
    print('loading data ...')
    np_load_old = np.load
    np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
    data = np.load(opt['load_path'] + 'joints_pca_etc.npz')

    keys = []
    vals = []
    for value in data['arr_0'].item():
        key = value
        values = data['arr_0'].item()[value]
        keys.append(key)
        vals.append(values)

    joints_pca = dict(zip(keys, vals))
    print('loaded')

    # run pose generator
    poses = generate_skeletons(joints_pca, opt)

    if opt['save']:
        with open(opt['save_path'] + 'generated_skeletons' + opt['save_suffix'] + '.json', 'w') as outfile:
            json.dump(poses, outfile)


if __name__ == '__main__':
    main()
