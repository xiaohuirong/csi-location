import numpy as np
from utils.parse_args import parse_args, show_args
from utils.cal_utils import turn_to_square, turn_back
import matplotlib.pyplot as plt

args = parse_args()
show_args(args)
r = args.round
s = args.scene

pos = np.load(args.output_pos_path)

t_pos = turn_to_square(r, s, pos)
c_t_pos = np.clip(t_pos, 0, 200)
t_c_t_pos = turn_back(r, s, c_t_pos)

fig, axes = plt.subplots(2, 2, figsize=(10, 10))
for i in range(2):
    for j in range(2):
        axes[i, j].set_aspect("equal", "box")
axes[0, 0].scatter(pos[:, 0], pos[:, 1], s=1)
axes[0, 1].scatter(t_pos[:, 0], t_pos[:, 1], s=1)
axes[1, 0].scatter(c_t_pos[:, 0], c_t_pos[:, 1], s=1)
axes[1, 1].scatter(t_c_t_pos[:, 0], t_c_t_pos[:, 1], s=1)

np.savetxt(args.txt_output_pos_path, t_c_t_pos, fmt="%.4f")

plt.show()
