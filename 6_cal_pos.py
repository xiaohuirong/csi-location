import torch
import numpy as np
from utils.parse_args import parse_args, show_args
from net.MLP import MLP, set_seed
from utils.cal_utils import turn_back, rotate_center_back

args = parse_args()
set_seed(args.tseed)

r = args.round
s = args.scene
m = args.method

dir = f"data/round{r}/s{s}/data/"
feature_dir = f"data/round{r}/s{s}/feature/"

result_dir = f"data/round{r}/s{s}/result/"
weight_path = result_dir + f"{m}:M{args.tseed}Round{r}Scene{s}.pth"

result_path = result_dir + f"{m}:Round{r}OutputPos{s}.npy"

test_feature_path = feature_dir + f"{m}:FRound{r}InputData{s}.npy"

cluster_index_path = dir + f"ClusterRound{r}Index{s}_S.npy"

aoa_path = feature_dir + f"Round{r}AoA{s}.npy"
if m == 5:
    clu_index = np.load(cluster_index_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_feature = np.load(test_feature_path)
if m == 5:
    test_feature = test_feature[clu_index]
if m == 11:
    pass
    # test_feature = test_feature[:, 408:]
test_feature = torch.from_numpy(test_feature).float().to(device)

# input_dim = 2 * 2 * 8 * 4 * 2 + 408
input_dim = test_feature.shape[1]

# defautl : 1024
embedding_dim = args.embedding

model = MLP(input_dim, embedding_dim).to(device)
model.load_state_dict(torch.load(weight_path))

show_args(args)

test_pre_pos = model(test_feature)

test_pre_pos = test_pre_pos.detach().cpu().numpy()

if m == 11:
    test_pre_pos = rotate_center_back(test_pre_pos, s, r)
    # aoa = np.load(aoa_path)
    # g0 = aoa > 0
    # test_pre_pos[g0, 0] = -test_pre_pos[g0, 0]
if m == 12:
    rho = test_pre_pos[:, 0]
    phi = test_pre_pos[:, 1] / 100 + np.pi / 6

    pre_x = rho * np.cos(phi)
    pre_y = rho * np.sin(phi)

    test_pre_pos = np.column_stack((pre_x, pre_y))

    test_pre_pos = rotate_center_back(test_pre_pos, s, r)


if args.turn:
    test_pre_pos = turn_back(r, s, test_pre_pos)

# np.savetxt(result_path, test_pre_pos.detach().cpu().numpy(), fmt="%.4f")
np.save(result_path, test_pre_pos)
