import numpy as np

from utils.parse_args import parse_args, show_args
from utils.cal_utils import cal_aoa

args = parse_args()
show_args(args)

r = args.round
s = args.scene
m = args.method
p = args.port
o = args.over


data_dir = f"data/round{r}/s{s}/data/"
feature_dir = f"data/round{r}/s{s}/feature/"

data_path = data_dir + f"Round{r}InputData{s}.npy"

aoa_path = feature_dir + f"Round{r}AoA{s}.npy"

all_aoa = []

H = np.load(data_path, mmap_mode="r")
bsz = H.shape[0]
batch = np.min([2000, H.shape[0]])
slice_num = bsz // batch

for i in range(slice_num):
    print(i)
    start = batch * i
    end = batch * (i + 1)

    # (2000, 2, 64, 408)
    h = np.fft.ifft(H[start:end])

    aoa = cal_aoa(h, r, s)

    if np.size(all_aoa) == 0:
        all_aoa = aoa
    else:
        all_aoa = np.concatenate((all_aoa, aoa), axis=0)

np.save(aoa_path, all_aoa)
