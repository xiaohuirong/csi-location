import numpy as np
from sklearn.cluster import AgglomerativeClustering
from utils.parse_args import parse_args, show_args
import matplotlib.pyplot as plt


args = parse_args()
show_args(args)

r = args.round
s = args.scene
m = args.method
p = args.port
o = args.over
f = args.filter


feature_dir = f"data/round{r}/s{s}/feature/"
data_dir = f"data/round{r}/s{s}/data/"

adpf_path = feature_dir + f"Port{p}Over{o}Filter{f}Adp{r}Scene{s}.npy"
label_path = feature_dir + f"Port{p}Over{o}Label{args.label}Round{r}Scene{s}.npy"
inputindex_s_path = data_dir + f"Round{r}Index{s}_S.npy"
inputpos_s_path = data_dir + f"Round{r}InputPos{s}_S.npy"

adp_f = np.load(adpf_path)
index = np.load(inputindex_s_path)
pos_s = np.load(inputpos_s_path)

label_num = args.label

# 使用层次聚类进行聚类
clustering = AgglomerativeClustering(
    n_clusters=label_num,  # 设定簇的数量
    metric="precomputed",  # 使用预计算的距离矩阵
    linkage="complete",  # 链接方法：'complete', 'average', 'single', 等
)


labels = clustering.fit_predict(adp_f)
np.save(label_path, labels)


all_all_l = [[] for _ in range(args.label)]
for i in range(len(labels)):
    label = labels[i]
    all_all_l[label].append(i)

nb_per_label_all = np.zeros(args.label)
for i in range(args.label):
    nb_per_label_all[i] = len(all_all_l[i])


labels_s = labels[index]
all_l = [[] for _ in range(args.label)]
for i in range(len(labels_s)):
    label = labels_s[i]
    all_l[label].append(i)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
nb_per_label = np.zeros(args.label)
for i in range(args.label):
    label_index = all_l[i]
    ax[0].scatter(pos_s[label_index, 0], pos_s[label_index, 1], label=f"{i}")
    nb_per_label[i] = len(label_index)

x = np.arange(args.label)
ax[1].bar(x, nb_per_label_all / 10, alpha=0.5, label="ALL")
ax[1].bar(x, nb_per_label, alpha=0.5, label="TEST")

g_10 = np.where(nb_per_label_all / 10 / nb_per_label > 2)[0]
print(g_10)

ax[0].set_aspect("equal", "box")
ax[0].legend()
ax[1].legend()
plt.show()
