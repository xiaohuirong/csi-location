import numpy as np
import matplotlib.pyplot as plt

from utils.parse_args import parse_args, show_args
from utils.plot_utils import get_color_map


def affine_transform_channel_chart(groundtruth_pos, channel_chart_pos, new_pre_pos):
    pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])
    unpad = lambda x: x[:, :-1]
    A, res, rank, s = np.linalg.lstsq(
        pad(channel_chart_pos), pad(groundtruth_pos), rcond=None
    )
    transform = lambda x: unpad(np.dot(pad(x), A))
    return transform(new_pre_pos)


args = parse_args()
show_args(args)

r = args.round
s = args.scene
m = args.method
l = args.label

data_dir = f"data/round{r}/s{s}/data/"
result_dir = f"data/round{r}/s{s}/result/"

if m == 5:
    cluster_index_path = data_dir + f"ClusterRound{r}Index{s}_S.npy"
    clu_index = np.load(cluster_index_path)

pre_pos_path = result_dir + f"{m}:Round{r}OutputPos{s}.txt"
pre_pos = np.loadtxt(pre_pos_path)

know_index_path = data_dir + f"Round{r}Index{s}_S.npy"
know_index = np.load(know_index_path)

if args.test:
    if m == 5:
        if r == 0:
            truth_pos_path = data_dir + f"Round{r}GroundTruth{s}.txt"
            truth_pos = np.loadtxt(truth_pos_path)

            truth_pos = truth_pos[clu_index]
            pre_pos = affine_transform_channel_chart(truth_pos, pre_pos, pre_pos)
            distance = np.linalg.norm(truth_pos - pre_pos, axis=-1)

            mean_distance = np.mean(distance)
            print(mean_distance)
        else:
            common_elements = np.intersect1d(know_index, clu_index)
            indices = np.where(np.isin(know_index, common_elements))[0]
            indices2 = np.where(np.isin(clu_index, common_elements))[0]

            know_pos_path = data_dir + f"Round{r}InputPos{s}_S.npy"
            know_pos = np.load(know_pos_path)
            pre_pos = affine_transform_channel_chart(
                know_pos[indices], pre_pos[indices2], pre_pos
            )
    else:
        if r == 0:
            truth_pos_path = data_dir + f"Round{r}GroundTruth{s}.txt"
            truth_pos = np.loadtxt(truth_pos_path)

            distance = np.linalg.norm(truth_pos - pre_pos, axis=-1)

            mean_distance = np.mean(distance)
            print(mean_distance)



color = get_color_map(pre_pos)

fig, axes = plt.subplots(1, 2, figsize=(20, 10))

axes[1].scatter(pre_pos[:, 0], pre_pos[:, 1], c=color, alpha=0.8, s=1)
if args.test:
    if r == 0:
        axes[0].scatter(truth_pos[:, 0], truth_pos[:, 1], c=color, alpha=0.8, s=1)
    else:
        axes[0].scatter(know_pos[indices, 0], know_pos[indices, 1], alpha=0.8, s=1)
        axes[1].scatter(pre_pos[indices2, 0], pre_pos[indices2, 1], alpha=0.8, s=1)
print(pre_pos.shape)

axes[0].set_aspect("equal", "box")
axes[1].set_aspect("equal", "box")
# axes[0].set_xlim(-180, -20)
# axes[0].set_ylim(-100, 150)
# axes[1].set_xlim(-180, -20)
# axes[1].set_ylim(-100, 150)
fig.tight_layout()
plt.show()
