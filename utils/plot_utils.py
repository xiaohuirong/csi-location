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


def get_color_map(positions):
    # Generate RGB colors for datapoints
    center_point = np.zeros(2, dtype=np.float32)
    center_point[0] = 0.5 * (
        np.min(positions[:, 0], axis=0) + np.max(positions[:, 0], axis=0)
    )
    center_point[1] = 0.5 * (
        np.min(positions[:, 1], axis=0) + np.max(positions[:, 1], axis=0)
    )

    def NormalizeData(in_data):
        return (in_data - np.min(in_data)) / (np.max(in_data) - np.min(in_data))

    rgb_values = np.zeros((positions.shape[0], 3))
    rgb_values[:, 0] = 1 - 0.9 * NormalizeData(positions[:, 0])
    rgb_values[:, 1] = 0.8 * NormalizeData(
        np.square(np.linalg.norm(positions - center_point, axis=1))
    )
    rgb_values[:, 2] = 0.9 * NormalizeData(positions[:, 1])

    return rgb_values


class DA:
    def __init__(self, dmg, known_pos_index, in_index, out_index):
        self.fig, self.axes = plt.subplots(2, 4, figsize=(20, 8))
        self.titles = [
            ["known", "whole"],
            ["in_in", "out_out"],
            ["in_out", "out_in"],
            ["in_know", "out_know"],
        ]

        for i in range(2):
            for j in range(4):
                self.axes[i, j].set_title(self.titles[j][i])
                self.axes[i, j].set_xlim(0, 350)
                self.axes[i, j].set_ylim(0, 0.008)

        self.axes[0, 0].hist(
            dmg[known_pos_index, :][:, known_pos_index].reshape(-1),
            bins=50,
            density=True,
        )
        self.axes[1, 0].hist(dmg.reshape(-1), bins=50, density=True)
        self.axes[0, 1].hist(
            dmg[in_index, :][:, in_index].reshape(-1), bins=50, density=True
        )
        self.axes[1, 1].hist(
            dmg[out_index, :][:, out_index].reshape(-1), bins=50, density=True
        )

        self.axes[0, 2].hist(
            dmg[in_index, :][:, out_index].reshape(-1), bins=50, density=True
        )
        self.axes[1, 2].hist(
            dmg[out_index, :][:, in_index].reshape(-1), bins=50, density=True
        )

        self.axes[0, 3].hist(
            dmg[in_index, :][:, known_pos_index].reshape(-1), bins=50, density=True
        )
        self.axes[1, 3].hist(
            dmg[out_index, :][:, known_pos_index].reshape(-1), bins=50, density=True
        )

    def plot(self):
        plt.figure(self.fig.number)  # 使用 plt.figure() 指定要显示的 fig 对象
        plt.tight_layout()
        plt.show()


def two_plot(pos_old, pos_new):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    colors = get_color_map(pos_old)

    axes[0].scatter(pos_old[:, 0], pos_old[:, 1], c=colors, alpha=0.8)
    axes[1].scatter(pos_new[:, 0], pos_new[:, 1], c=colors, alpha=0.8)

    axes[0].set_aspect("equal", "box")
    axes[1].set_aspect("equal", "box")

    plt.show()


def pos_plot(poses):
    fig, axes = plt.subplots(figsize=(5, 5))
    i = 0
    for pos in poses:
        axes.scatter(pos[:, 0], pos[:, 1], alpha=0.8, label=f"{i}")
        i += 1
        print(np.min(pos[:, 0]))
    axes.set_aspect("equal", "box")
    axes.legend()
    plt.show()
