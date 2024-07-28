import numpy as np

from utils.parse_args import parse_args, show_args
from utils.cal_utils import cal_aoa_tof

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
tof_path = feature_dir + f"Round{r}ToF{s}.npy"
aop_path = feature_dir + f"Round{r}AoP{s}.npy"
max_index_path = feature_dir + f"Round{r}MaxIndex{s}.npy"
h_path = feature_dir + f"Round{r}Time{s}.npy"

all_aoa = []
all_tof = []
all_aop = []
all_max_index = []
all_h = []

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

    aoa, tof, aop, max_index = cal_aoa_tof(h, r, s)

    if np.size(all_aoa) == 0:
        all_aoa = aoa
        all_tof = tof
        all_aop = aop
        all_max_index = max_index
        all_h = h[np.arange(batch), :, :, max_index]
    else:
        all_aoa = np.concatenate((all_aoa, aoa), axis=0)
        all_tof = np.concatenate((all_tof, tof), axis=0)
        all_aop = np.concatenate((all_aop, aop), axis=0)
        all_max_index = np.concatenate((all_max_index, max_index), axis=0)
        all_h = np.concatenate((all_h, h[np.arange(batch), :, :, max_index]), axis=0)
    print(all_h.shape)

np.save(aoa_path, all_aoa)
np.save(tof_path, all_tof)
np.save(aop_path, all_aop)
np.save(max_index_path, all_max_index)
np.save(h_path, all_h)
