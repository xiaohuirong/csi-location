import numpy as np
from utils.parse_args import parse_args, show_args
import matplotlib.pyplot as plt
from utils.cal_utils import rotate_points, remap1


args = parse_args()
show_args(args)

r = args.round
s = args.scene
m = args.method
l = args.label


data_dir = f"data/round{r}/s{s}/data/"
feature_dir = f"data/round{r}/s{s}/feature/"
result_dir = f"data/round{r}/s{s}/result/"

# pos_path = result_dir + f"{m}:Round{r}OutputPos{s}.txt"
pos_path = "/home/xiao/Test/train_data/chusai/data/round1/result_best/fix1_change_1_new/Round1OutputPos3.txt"
cluster_index_path = data_dir + f"ClusterRound{r}Index{s}_S.npy"

pos = np.loadtxt(pos_path)
out_index = np.load(cluster_index_path)

in_index = np.arange(20000)
in_index = in_index[~np.isin(in_index, out_index)]
print(in_index.shape)

fig, axes = plt.subplots(figsize=(5, 5))

pos[out_index, :] = rotate_points(pos[out_index, :], -15)
# pos[out_index, :] = remap1(pos[out_index, :])

axes.scatter(pos[in_index, 0], pos[in_index, 1], s=1)
axes.scatter(pos[out_index, 0], pos[out_index, 1], s=1)

axes.set_aspect("equal", "box")
plt.show()

np.savetxt("data/rotate.txt", pos, fmt="%.4f")
