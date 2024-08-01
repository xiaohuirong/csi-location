import numpy as np

from utils.parse_args import parse_args, show_args
from utils.cal_utils import cal_aoa_tof_feature

args = parse_args()
show_args(args)

r = args.round
s = args.scene

if args.test:
    input_path = args.test_data_slice_path
    save_path = args.test_feature_slice_path
else:
    if args.slice:
        inputdata_path = args.data_slice_path
        save_path = args.feature_slice_path
    else:
        input_path = args.data_path
        save_path = args.feature_path

all_feature = []

H = np.load(input_path, mmap_mode="r")
bsz = H.shape[0]
batch = np.min([2000, H.shape[0]])
slice_num = bsz // batch
for i in range(slice_num):
    print(i)
    start = batch * i
    end = batch * (i + 1)

    # (2000, 2, 64, 408)
    h = np.fft.ifft(H[start:end])

    power = np.square(np.abs(h))

    # (2000, 408)
    sum_power = np.mean(power, axis=(1, 2))

    max_indexs = np.argmax(sum_power, axis=-1)

    pre_dis = max_indexs * 299.792458 / (408 * 0.24)
    pre_dis = pre_dis.reshape(pre_dis.shape[0], 1)

    # (2000, 2, 64)
    h_max = h[np.arange(batch), :, :, max_indexs]
    h_max = h_max.reshape(batch, 2, 2, 8, 4)

    h_max_conj = np.conj(h_max).reshape(batch, 2, 2, 32, 1)
    h_max = h_max.reshape(batch, 2, 2, 1, 32)

    h_diff = h_max * h_max_conj
    h_diff = h_diff / np.abs(h_diff)

    h_diff = h_diff.reshape(batch, 2 * 2 * 32 * 32)

    h_diff_real = h_diff.real
    h_diff_imag = h_diff.imag

    feature_new = cal_aoa_tof_feature(h, r, s)
    feature = np.concatenate((sum_power, h_diff_real, h_diff_imag), axis=-1)
    feature = np.concatenate((feature_new, feature), axis=-1)

    if np.size(all_feature) == 0:
        all_feature = feature
    else:
        all_feature = np.concatenate((all_feature, feature), axis=0)

np.save(save_path, all_feature)
