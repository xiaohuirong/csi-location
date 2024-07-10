import numpy as np

import itertools

from utils.parse_args import parse_args, show_args

args = parse_args()
show_args(args)

r = args.round
s = args.scene

dir = f"data/round{r}/s{s}/data/"

cfg_path = dir + f"Round{r}CfgData{s}.txt"
inputdata_path = dir + f"Round{r}InputData{s}.txt"


# 切片读取函数
def read_slice_of_file(file_path, start, end):
    with open(file_path, "r") as file:
        # 使用itertools.islice进行切片处理

        slice_lines = list(itertools.islice(file, start, end))

    return slice_lines


# 读取RoundYCfgDataX.txt中样本数信息

slice_lines = read_slice_of_file(cfg_path, 1, 6)

info = np.loadtxt(slice_lines)

# tol_samp_num : 20000
tol_samp_num = int(info[0])

# port_num : 2
port_num = int(info[2])

# ant_num : 64
ant_num = int(info[3])

# sc_num : 408
sc_num = int(info[4])


# 切片读取RoundYInputDataX.txt信道信息

H = []

slice_samp_num = 1000  # 切片样本数量

slice_num = int(tol_samp_num / slice_samp_num)  # 切片数量

for slice_idx in range(slice_num):
    print(slice_idx)

    slice_lines = read_slice_of_file(
        inputdata_path, slice_idx * slice_samp_num, (slice_idx + 1) * slice_samp_num
    )

    Htmp = np.loadtxt(slice_lines).astype(np.float32)

    Htmp = np.reshape(Htmp, (slice_samp_num, 2, sc_num, ant_num, port_num))

    Htmp = Htmp[:, 0, :, :, :] + 1j * Htmp[:, 1, :, :, :]
    Htmp = np.transpose(Htmp, (0, 3, 2, 1))

    if np.size(H) == 0:
        H = Htmp

    else:
        H = np.concatenate((H, Htmp), axis=0)

save_path = dir + f"Round{r}InputData{s}.npy"
# H (20000, 2, 64, 408)
np.save(save_path, H)
