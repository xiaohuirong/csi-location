import torch
import numpy as np
from utils.parse_args import parse_args, show_args
from net.MLP import MLP, set_seed

args = parse_args()
set_seed(args.tseed)

r = args.round
s = args.scene
m = args.method

dir = f"data/round{r}/s{s}/data/"
feature_dir = f"data/round{r}/s{s}/feature/"

result_dir = f"data/round{r}/s{s}/result/"
weight_path = result_dir + f"{m}:M{args.tseed}Round{r}Scene{s}.pth"

result_path = result_dir + f"{m}:Round{r}OutputPos{s}.txt"

test_feature_path = feature_dir + f"{m}:FRound{r}InputData{s}.npy"

cluster_index_path = dir + f"ClusterRound{r}Index{s}_S.npy"
clu_index = np.load(cluster_index_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_feature = np.load(test_feature_path)
if m == 5:
    test_feature = test_feature[clu_index]
test_feature = torch.from_numpy(test_feature).float().to(device)

# input_dim = 2 * 2 * 8 * 4 * 2 + 408
input_dim = test_feature.shape[1]

# defautl : 1024
embedding_dim = args.embedding

model = MLP(input_dim, embedding_dim).to(device)
model.load_state_dict(torch.load(weight_path))

show_args(args)

test_pre_pos = model(test_feature)
np.savetxt(result_path, test_pre_pos.detach().cpu().numpy(), fmt="%.4f")
