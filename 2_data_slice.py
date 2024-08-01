import numpy as np
from utils.parse_args import parse_args, show_args
import os

args = parse_args()
show_args(args)

r = args.round
s = args.scene

if not os.path.exists(args.data_path):
    print(f"The file {args.data_path} does not exist.")
    exit()

# H (bsz, port_num, ant_num, sc_num)
H = np.load(args.data_path, mmap_mode="r")

input_pos = np.loadtxt(args.txt_pos_path)
indexes = input_pos[:, 0].astype(int) - 1
pos = input_pos[:, 1:]
h = H[indexes]

np.save(args.data_slice_path, h)
np.save(args.pos_path, pos)
np.save(args.index_path, indexes)
