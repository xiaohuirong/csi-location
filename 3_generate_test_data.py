import numpy as np
from utils.parse_args import parse_args, show_args
import os

args = parse_args()
show_args(args)

np.random.seed(args.seed)

low = 0
high = 19999
sample_size = 2000
sample = np.random.choice(range(low, high + 1), sample_size, replace=False)

r = args.round
s = args.scene

if not os.path.exists(args.data_path):
    print(f"The file {args.data_path} does not exist.")
    exit()

if not os.path.exists(args.truth_pos_path):
    print(f"The file {args.truth_pos_path} does not exist.")
    exit()

# H (bsz, port_num, ant_num, sc_num)
H = np.load(args.data_path, mmap_mode="r")

truth_pos = np.loadtxt(args.truthpos_path)

h = H[sample]

pos = truth_pos[sample]

np.save(args.test_data_slice_path, h)
np.save(args.test_pos_path, pos)
np.save(args.test_index_path, sample)
