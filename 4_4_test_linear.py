from utils.parse_args import parse_args, show_args
import matplotlib.pyplot as plt
import numpy as np
from utils.plot_utils import plot_dissimilarity_over_euclidean_distance

args = parse_args()
show_args(args)

r = args.round
s = args.scene
m = args.method
p = args.port
o = args.over

if not args.test:
    print("Not specilize --test")
    exit()

dmg_path = f"data/round{r}/s{s}/feature/Port{p}Over{o}Dmg{r}Scene{s}.npy"
dis_path = f"data/round{r}/s{s}/feature/Dis{r}Scene{s}.npy"

index_path = f"data/round{r}/s{s}/data/Round{r}InputPos{s}.txt"

index = np.loadtxt(index_path)[:, 0].astype(np.int32) - 1

dis = np.load(dis_path)
dmg = np.load(dmg_path)

plt.figure(figsize=(8, 4))

plot_dissimilarity_over_euclidean_distance(
    dmg[index, :][:, index], dis[index, :][:, index], "Dmg"
)

plt.legend()
plt.xlabel("Euclidean Distance [m]")
plt.ylabel("Channel distance")
plt.show()
