import matplotlib.pyplot as plt
import numpy as np

adp_dissimilarity_matrix = np.load("data/Round0InputData1_S1_ADP.npy")
# adp_dissimilarity_matrix = np.load("data/adp_need.npy")

print(adp_dissimilarity_matrix[0, 0:10])

# exit()

input_pos = np.loadtxt("data/Round0InputPos1.txt")
# input_pos = np.load("data/pos.npy")
print(input_pos.shape)

bsz = adp_dissimilarity_matrix.shape[0]

index = 199

pos = input_pos[:, 1:]
x = input_pos[:, 1]
y = input_pos[:, 2]

# distance = np.linalg.norm(pos.reshape(bsz, 1, 2) - pos.reshape(1, bsz, 2), axis=2)

weights = adp_dissimilarity_matrix[index]
# weights = np.clip(weights, 119.99, 120)
# weights = distance[index]
# weights = np.max(np.abs(csi_time_domain), axis=(1, 2, 3))
print(weights)

# 设置画布和子图
fig, (ax_scatter, ax_hist) = plt.subplots(1, 2, figsize=(12, 6))

# 绘制散点图，根据权重使用viridis色彩映射
scatter = ax_scatter.scatter(x, y, c=weights, cmap="viridis", s=50, alpha=0.8)
ax_scatter.scatter(x[index], y[index], color="red")
ax_scatter.set_xlabel("X")
ax_scatter.set_ylabel("Y")
ax_scatter.set_title("Scatter Plot with Weighted Colors")

# 添加权重颜色条
cbar = fig.colorbar(scatter, ax=ax_scatter, label="Weights")

# 绘制权重的直方图
ax_hist.hist(weights, bins=10, alpha=0.6, color="gray", edgecolor="black")
ax_hist.set_xlabel("Weights")
ax_hist.set_ylabel("Frequency")
ax_hist.set_title("Histogram of Weights")

plt.tight_layout()
plt.show()
