#!/bin/python

import numpy as np
import matplotlib.pyplot as plt


def plot_colorized(positions, groundtruth_positions, title=None, show=True, alpha=1.0):
    # Generate RGB colors for datapoints
    center_point = np.zeros(2, dtype=np.float32)
    center_point[0] = 0.5 * (
        np.min(groundtruth_positions[:, 0], axis=0)
        + np.max(groundtruth_positions[:, 0], axis=0)
    )
    center_point[1] = 0.5 * (
        np.min(groundtruth_positions[:, 1], axis=0)
        + np.max(groundtruth_positions[:, 1], axis=0)
    )

    def NormalizeData(in_data):
        return (in_data - np.min(in_data)) / (np.max(in_data) - np.min(in_data))

    rgb_values = np.zeros((groundtruth_positions.shape[0], 3))
    rgb_values[:, 0] = 1 - 0.9 * NormalizeData(groundtruth_positions[:, 0])
    rgb_values[:, 1] = 0.8 * NormalizeData(
        np.square(np.linalg.norm(groundtruth_positions - center_point, axis=1))
    )
    rgb_values[:, 2] = 0.9 * NormalizeData(groundtruth_positions[:, 1])

    # Plot datapoints
    plt.figure(figsize=(6, 6))
    if title is not None:
        plt.title(title, fontsize=16)
    plt.scatter(
        positions[:, 0], positions[:, 1], c=rgb_values, alpha=alpha, s=10, linewidths=0
    )
    plt.xlabel("x coordinate")
    plt.ylabel("y coordinate")
    if show:
        plt.show()


pos_index = np.loadtxt("data/Round0InputPos1.txt")
index = pos_index[:, 0]
pos = pos_index[:, 1:]

# pos = np.loadtxt("data/Round0GroundTruth1.txt")

plot_colorized(pos, pos, title="Ground Truth Positions")
