import numpy as np
from utils.parse_args import parse_args, show_args

args = parse_args()
show_args(args)

r = args.round
s = args.scene
m = args.method

if not args.test:
    print("Not specilize --test")
    exit()

pos_path = f"data/round{r}/s{s}/data/Round{r}GroundTruth{s}.txt"
dis_path = f"data/round{r}/s{s}/feature/Dis{r}Scene{s}.npy"


pos = np.loadtxt(pos_path).astype(np.float32)

bsz = pos.shape[0]

pos = pos.reshape(1, bsz, 2)

pos_r = pos.reshape(bsz, 1, 2)

dis = np.linalg.norm(pos - pos_r, axis=-1)

np.save(dis_path, dis)
