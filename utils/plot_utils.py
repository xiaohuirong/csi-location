import matplotlib.pyplot as plt
import numpy as np


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
            bin_25_perc[i - 1], bin_medians[i - 1], bin_75_perc[i - 1] = (
                bin_25_perc[i - 2],
                bin_medians[i - 2],
                bin_75_perc[i - 2],
            )

    plt.plot(bins[:-1], bin_medians, label=label)
    plt.fill_between(bins[:-1], bin_25_perc, bin_75_perc, alpha=0.5)
