import numpy as np
import matplotlib.pyplot as plt

from utils.parse_args import parse_args, show_args

args = parse_args()
show_args(args)

r = args.round
s = args.scene

data_dir = f"data/round{r}/s{s}/data/"
result_dir = f"data/round{r}/s{s}/result/"

truth_pos_path = data_dir + f"Round{r}GroundTruth{s}.txt"
pre_pos_path = result_dir + f"Round{r}OutputPos{s}.txt"


truth_pos = np.loadtxt(truth_pos_path)
pre_pos = np.loadtxt(pre_pos_path)


distance = np.linalg.norm(truth_pos - pre_pos, axis=-1)

mean_distance = np.mean(distance)

fig, axes = plt.subplots(1, 2, figsize=(20, 10))

axes[0].scatter(truth_pos[:, 0], truth_pos[:, 1])
axes[1].scatter(pre_pos[:, 0], pre_pos[:, 1])

print(mean_distance)

plt.show()
