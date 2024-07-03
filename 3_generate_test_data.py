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

dir = f"data/round{r}/s{s}/data/"

inputdata_path = dir + f"Round{r}InputData{s}.npy"
truthpos_path = dir + f"Round{r}GroundTruth{s}.txt"

if not os.path.exists(inputdata_path):
    print(f"The file {inputdata_path} does not exist.")
    exit()

if not os.path.exists(truthpos_path):
    print(f"The file {truthpos_path} does not exist.")
    exit()

# H (bsz, port_num, ant_num, sc_num)
H = np.load(inputdata_path, mmap_mode="r")

truth_pos = np.loadtxt(truthpos_path)

h = H[sample]

pos = truth_pos[sample]

test_s_data = dir + f"Test{args.seed}Round{r}InputData{s}_S.npy"
test_s_pos = dir + f"Test{args.seed}Round{r}InputPos{s}_S.npy"
test_s_index = dir + f"Test{args.seed}Round{r}Index{s}_S.npy"

np.save(test_s_data, h)
np.save(test_s_pos, pos)
np.save(test_s_index, sample)
