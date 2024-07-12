import numpy as np
from utils.parse_args import parse_args, show_args


args = parse_args()
show_args(args)

r = args.round
s = args.scene
m = args.method
p = args.port
o = args.over


data_path = f"data/round{r}/s{s}/data/Round{r}InputData{s}.npy"
data_part_path = f"data/round{r}/s{s}/data/Port{p}Over{o}Round{r}InputData{s}"

H = np.load(data_path, mmap_mode="r")
[bsz, port_num, ant_num, sr_num] = H.shape

H = H.reshape(bsz, port_num, 2, ant_num // 2, sr_num)

h = np.fft.ifft(H[:, p, o, :, :])[:, :, 0:100]

np.save(data_part_path, h.astype(np.complex64))
