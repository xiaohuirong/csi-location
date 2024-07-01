import matplotlib.pyplot as plt


def show_distance(matrix, pos, index):
    """
    matrix: (bsz, bsz)
    pos: (bsz, 2)
    index: int
    """
    weights = matrix[index]

    # 设置画布和子图
    fig, (ax_scatter, ax_hist) = plt.subplots(1, 2, figsize=(12, 6))

    # 绘制散点图，根据权重使用viridis色彩映射
    scatter = ax_scatter.scatter(
        pos[:, 0], pos[:, 1], c=weights, cmap="viridis", s=50, alpha=0.8
    )
    ax_scatter.scatter(pos[index, 0], pos[index, 1], color="red")
    ax_scatter.set_xlabel("X")
    ax_scatter.set_ylabel("Y")
    ax_scatter.set_title("Scatter Plot with Weighted Colors")

    # 添加权重颜色条
    fig.colorbar(scatter, ax=ax_scatter, label="Weights")

    # 绘制权重的直方图
    ax_hist.hist(weights, bins=10, alpha=0.6, color="gray", edgecolor="black")
    ax_hist.set_xlabel("Weights")
    ax_hist.set_ylabel("Frequency")
    ax_hist.set_title("Histogram of Weights")

    plt.tight_layout()
    plt.show()
