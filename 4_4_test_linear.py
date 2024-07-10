from utils.parse_args import parse_args, show_args
import matplotlib.pyplot as plt
import numpy as np

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

dis = np.load(dis_path)
dmg = np.load(dmg_path)


def plot_dissimilarity_over_euclidean_distance(
    dissimilarity_matrix, distance_matrix, label=None
):
    nth_reduction = 10
    dissimilarities_flat = dissimilarity_matrix[
        ::nth_reduction, ::nth_reduction
    ].flatten()
    distances_flat = distance_matrix[::nth_reduction, ::nth_reduction].flatten()

    max_distance = np.max(distances_flat)
    bins = np.linspace(0, max_distance, 200)
    bin_indices = np.digitize(distances_flat, bins)

    bin_medians = np.zeros(len(bins) - 1)
    bin_25_perc = np.zeros(len(bins) - 1)
    bin_75_perc = np.zeros(len(bins) - 1)
    for i in range(1, len(bins)):
        bin_values = dissimilarities_flat[bin_indices == i]
        try:
            bin_25_perc[i - 1], bin_medians[i - 1], bin_75_perc[i - 1] = np.percentile(
                bin_values, [25, 50, 75]
            )
        except:
            print("missing")

    plt.plot(bins[:-1], bin_medians, label=label)
    plt.fill_between(bins[:-1], bin_25_perc, bin_75_perc, alpha=0.5)


plt.figure(figsize=(8, 4))

plot_dissimilarity_over_euclidean_distance(dmg, dis, "Dmg")

plt.legend()
plt.xlabel("Euclidean Distance [m]")
plt.ylabel("Channel distance")
plt.show()
