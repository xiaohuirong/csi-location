import numpy as np
from utils.parse_args import parse_args, show_args

args = parse_args()
show_args(args)

r = args.round
s = args.scene
m = args.method


result_dir = f"data/round{r}/s{s}/result/"
data_dir = f"data/round{r}/s{s}/data/"

input_path = data_dir + f"Round{r}InputPos{s}.txt"
output_path = result_dir + f"{m}:Round{r}OutputPos{s}.txt"

output_pos = np.loadtxt(output_path)
input_pos = np.loadtxt(input_path)

index = input_pos[:, 0].astype(int) - 1
input_pos = input_pos[:, 1:]

diff_pos = input_pos - output_pos[index]

norm = np.linalg.norm(diff_pos, axis=-1)

fill_pos = output_pos
fill_pos[index] = input_pos

np.savetxt(output_path, fill_pos, fmt="%.4f")

print(np.sum(norm) / 60000)
