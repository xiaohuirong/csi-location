import numpy as np
import matplotlib.pyplot as plt

from utils.parse_args import parse_args, show_args


def get_color_map(positions):
    # Generate RGB colors for datapoints
    center_point = np.zeros(2, dtype=np.float32)
    center_point[0] = 0.5 * (
        np.min(positions[:, 0], axis=0) + np.max(positions[:, 0], axis=0)
    )
    center_point[1] = 0.5 * (
        np.min(positions[:, 1], axis=0) + np.max(positions[:, 1], axis=0)
    )

    def NormalizeData(in_data):
        return (in_data - np.min(in_data)) / (np.max(in_data) - np.min(in_data))

    rgb_values = np.zeros((positions.shape[0], 3))
    rgb_values[:, 0] = 1 - 0.9 * NormalizeData(positions[:, 0])
    rgb_values[:, 1] = 0.8 * NormalizeData(
        np.square(np.linalg.norm(positions - center_point, axis=1))
    )
    rgb_values[:, 2] = 0.9 * NormalizeData(positions[:, 1])

    return rgb_values


args = parse_args()
show_args(args)

r = args.round
s = args.scene
m = args.method

data_dir = f"data/round{r}/s{s}/data/"
result_dir = f"data/round{r}/s{s}/result/"

cluster_index_path = data_dir + f"ClusterRound{r}Index{s}_S.npy"
clu_index = np.load(cluster_index_path)

pre_pos_path = result_dir + f"{m}:Round{r}OutputPos{s}.txt"
pre_pos = np.loadtxt(pre_pos_path)

if args.test:
    truth_pos_path = data_dir + f"Round{r}GroundTruth{s}.txt"
    truth_pos = np.loadtxt(truth_pos_path)

    if m == 5:
        truth_pos = truth_pos[clu_index]

    distance = np.linalg.norm(truth_pos - pre_pos, axis=-1)

    mean_distance = np.mean(distance)
    print(mean_distance)

color = get_color_map(pre_pos)

fig, axes = plt.subplots(1, 2, figsize=(20, 10))

if args.test:
    axes[0].scatter(truth_pos[:, 0], truth_pos[:, 1], c=color, alpha=0.8)
axes[1].scatter(pre_pos[:, 0], pre_pos[:, 1], c=color, alpha=0.8)

axes[0].set_aspect("equal", "box")
axes[1].set_aspect("equal", "box")
fig.tight_layout()
plt.show()
