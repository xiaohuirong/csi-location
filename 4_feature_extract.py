import numpy as np

from utils.parse_args import parse_args, show_args

args = parse_args()
show_args(args)

r = args.round
s = args.scene

if args.data == "None":
    print("Please set --data argument to select data file.")
    exit()

dir = f"data/round{r}/s{s}/data/"
inputdata_path = dir + args.data
feature_dir = f"data/round{r}/s{s}/feature/"

H = np.load(inputdata_path, mmap_mode="r")

bsz = H.shape[0]

slice_num = bsz // 2000

all_feature = []

for i in range(slice_num):
    print(i)
    start = 2000 * i
    end = 2000 * (i + 1)

    # (2000, 2, 64, 408)
    h = np.fft.ifft(H[start:end])

    power = np.square(np.abs(h))

    # (2000, 408)
    sum_power = np.sum(power, axis=(1, 2))

    max_indexs = np.argmax(sum_power, axis=-1)

    # (2000, 2, 64)
    h_max = h[np.arange(2000), :, :, max_indexs]
    h_max = h_max.reshape(2000, 2, 2, 8, 4)

    # 第一个天线
    h_max_conj = np.conj(h_max[:, :, :, 0, :]).reshape(2000, 2, 2, 1, 4)

    h_diff = h_max * h_max_conj
    h_diff = h_diff / np.abs(h_diff)

    h_diff = h_diff.reshape(2000, 2 * 2 * 8 * 4)

    h_diff_real = h_diff.real
    h_diff_imag = h_diff.imag

    feature = np.concatenate((sum_power, h_diff_real, h_diff_imag), axis=-1)
    if np.size(all_feature) == 0:
        all_feature = feature
    else:
        all_feature = np.concatenate((all_feature, feature), axis=0)

save_path = feature_dir + "F" + args.data
np.save(save_path, all_feature)
