import numpy as np
from utils.parse_args import parse_args, show_args
import ast


args = parse_args()
show_args(args)

r = args.round
s = args.scene
m = args.method
p = args.port
o = args.over
f = args.filter


data_dir = f"data/round{r}/s{s}/data/"
feature_dir = f"data/round{r}/s{s}/feature/"
label_path = feature_dir + f"Port{p}Over{o}Label{args.label}Round{r}Scene{s}.npy"
cluster_data_path = data_dir + f"ClusterRound{r}InputData{s}_S.npy"
cluster_index_path = data_dir + f"ClusterRound{r}Index{s}_S.npy"

all_data_path = data_dir + f"Round{r}InputData{s}.npy"

all_data = np.load(all_data_path).astype(np.complex64)

clusters = ast.literal_eval(args.clusters)
print(clusters)
labels = np.load(label_path)

indices = []
for i in range(len(labels)):
    label = labels[i]
    if label in clusters:
        indices.append(i)
print(len(indices))

cluster_data = all_data[indices]

np.save(cluster_data_path, cluster_data)
np.save(cluster_index_path, indices)
