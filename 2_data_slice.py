import numpy as np
from utils.parse_args import parse_args, show_args
import os

args = parse_args()
show_args(args)

r = args.round
s = args.scene

dir = f"data/round{r}/s{s}/data/"

inputdata_path = dir + f"Round{r}InputData{s}.npy"
inputpos_path = dir + f"Round{r}InputPos{s}.txt"

if not os.path.exists(inputdata_path):
    print(f"The file {inputdata_path} does not exist.")
    exit()

# H (bsz, port_num, ant_num, sc_num)
H = np.load(inputdata_path, mmap_mode="r")

input_pos = np.loadtxt(inputpos_path)
indexes = input_pos[:, 0].astype(int) - 1
pos = input_pos[:, 1:]
h = H[indexes]

inputdata_s_path = dir + f"Round{r}InputData{s}_S.npy"
inputpos_s_path = dir + f"Round{r}InputPos{s}_S.npy"
inputindex_s_path = dir + f"Round{r}Index{s}_S.npy"
np.save(inputdata_s_path, h)
np.save(inputpos_s_path, pos)
np.save(inputindex_s_path, indexes)
