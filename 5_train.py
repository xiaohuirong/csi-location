import torch
from torch.utils.data import DataLoader
import numpy as np
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from utils.parse_args import parse_args, show_args
import tqdm
from net.MLP import (
    CustomDataset,
    MLP,
    set_seed,
    CustomDataset3,
    cal_loss,
    CustomDataLoader,
)
from utils.cal_utils import turn_to_square, rotate_center_to_y, rotate_center_back
from utils.plot_utils import show_pos

args = parse_args()
set_seed(args.tseed)

r = args.round
s = args.scene
m = args.method
p = args.port
o = args.over

dir = f"data/round{r}/s{s}/data/"
feature_dir = f"data/round{r}/s{s}/feature/"

cluster_index_path = dir + f"ClusterRound{r}Index{s}_S.npy"
if m == 5:
    clu_index = np.load(cluster_index_path)

if m == 4 or m == 5:
    feature_path = feature_dir + f"{m}:FRound{r}InputData{s}.npy"
else:
    feature_path = feature_dir + f"{m}:FRound{r}InputData{s}_S.npy"
pos_path = dir + f"Round{r}InputPos{s}_S.npy"

test_feature_path = feature_dir + f"{m}:FTest{args.seed}Round{r}InputData{s}_S.npy"
test_pos_path = dir + f"Test{args.seed}Round{r}InputPos{s}_S.npy"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


pos = np.load(pos_path)

if m == 11:
    pos = rotate_center_to_y(pos, s, r)
    if s == 3:
        s2 = 1
        dir2 = f"data/round{r}/s{s2}/data/"
        pos_path2 = dir2 + f"Round{r}InputPos{s2}_S.npy"
        pos2 = np.load(pos_path2)
        pos2 = rotate_center_to_y(pos2, s2, r)
        pos = np.vstack((pos, pos2))
    elif s == 6:
        s2 = 4
        dir2 = f"data/round{r}/s{s2}/data/"
        pos_path2 = dir2 + f"Round{r}InputPos{s2}_S.npy"
        pos2 = np.load(pos_path2)
        pos2 = rotate_center_to_y(pos2, s2, r)
        pos = np.vstack((pos, pos2))

    # show_pos(pos)
    # show_pos(rotate_center_back(pos, s, r))

feature = np.load(feature_path)
# feature = feature[:, 408:]
if m == 5:
    feature = feature[clu_index]

if m == 11:
    if s == 3:
        s2 = 1
        feature_dir2 = f"data/round{r}/s{s2}/feature/"
        feature_path2 = feature_dir2 + f"{m}:FRound{r}InputData{s2}_S.npy"
        feature2 = np.load(feature_path2)
        feature = np.vstack((feature, feature2))
    elif s == 6:
        s2 = 4
        feature_dir2 = f"data/round{r}/s{s2}/feature/"
        feature_path2 = feature_dir2 + f"{m}:FRound{r}InputData{s2}_S.npy"
        feature2 = np.load(feature_path2)
        feature = np.vstack((feature, feature2))

feature = torch.from_numpy(feature).float().to(device)

if m == 12:
    pos = rotate_center_to_y(pos, s, r)
    rho = np.linalg.norm(pos, axis=-1)
    phi = (np.arctan2(pos[:, 1], pos[:, 0]) - np.pi / 6) * 100
    pos = np.column_stack((rho, phi))


if args.turn:
    pos = turn_to_square(r, s, pos)
pos = torch.from_numpy(pos).float().to(device)

if args.test:
    test_feature = np.load(test_feature_path)
    # test_feature = test_feature[:, 408:]
    test_pos = np.load(test_pos_path)
    if args.turn:
        test_pos = turn_to_square(r, s, test_pos)
    test_feature = torch.from_numpy(test_feature).float().to(device)
    test_pos = torch.from_numpy(test_pos).float().to(device)

if m == 3:
    dataset = CustomDataset3(feature, pos)
    if args.test:
        test_dataset = CustomDataset3(test_feature, test_pos)
else:
    dataset = CustomDataset(feature, pos)
    if args.test:
        test_dataset = CustomDataset(test_feature, test_pos)

# default : 100
batch_size = args.bsz

# input_dim = 2 * 2 * 8 * 4 * 2 + 408
input_dim = feature.shape[1]

# defautl : 1024
embedding_dim = args.embedding

k = 100

# defautl 1e-3
lr = args.lr
# default 200
step = args.step
# default 0.9
gamma = args.gamma
if m == 3:
    model = MLP(input_dim // k * 3, embedding_dim).to(device)
else:
    model = MLP(input_dim, embedding_dim).to(device)

if args.cp != "None":
    model.load_state_dict(torch.load(args.cp))

result_dir = f"data/round{r}/s{s}/result/"
result_path = result_dir + f"{m}:M{args.tseed}Round{r}Scene{s}.pth"
if args.load:
    model.load_state_dict(torch.load(result_path))

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
scheduler = lr_scheduler.StepLR(optimizer, step_size=step, gamma=gamma)

writer = SummaryWriter(f"runs/R{r}S{s}M{m}-{args.time}")
writer.add_text(
    "parameter",
    "|param|value|\n|-|-|\n%s"
    % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
)

show_args(args)
# default : 5000
epoch_num = args.epoch

if args.test:
    test_dataloader = DataLoader(test_dataset, batch_size=2000, shuffle=False)

if args.method == 4 or m == 5:
    dmg_path = f"data/round{r}/s{s}/feature/Port{p}Over{o}Dmg{r}Scene{s}.npy"
    dmg = np.load(dmg_path)
    if m == 5:
        dmg = dmg[clu_index, :][:, clu_index]
    dmg = torch.from_numpy(dmg).to(device)

with tqdm.tqdm(total=epoch_num) as bar:
    for epoch in range(epoch_num):
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        all_loss = []
        for batch in dataloader:
            if m == 3:
                feature = batch["feature"].reshape(batch_size * k, input_dim // k * 3)
                pos = batch["pos"].repeat_interleave(p, dim=0)
            else:
                feature = batch["feature"]
                pos = batch["pos"]
                index = batch["index"]

            pre_pos = model(feature)

            # if args.turn:
            #     pre_pos = torch.clip(pre_pos, 0, 200)

            if m == 4 or m == 5:
                loss = cal_loss(pre_pos, dmg[index, :][:, index])
            elif m == 12:
                diff = torch.square(pos - pre_pos)
                loss = torch.mean(diff)
            else:
                diff = torch.norm(pos - pre_pos, dim=-1)
                loss = torch.mean(diff)

            all_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        mean_all_loss = torch.mean(torch.tensor(all_loss))
        # print(f"train loss: {mean_all_loss.item()}")
        writer.add_scalar("rate/train_loss", mean_all_loss.item(), epoch)

        scheduler.step()

        if epoch % 100 == 0 and args.test:
            for test_batch in test_dataloader:
                if m == 3:
                    test_feature = test_batch["feature"].reshape(
                        2000 * k, input_dim // k * 3
                    )
                    test_pos = test_batch["pos"].repeat_interleave(k, dim=0)
                else:
                    test_feature = test_batch["feature"]
                    test_pos = test_batch["pos"]
                test_pre_pos = model(test_feature)
                test_diff = torch.norm(test_pos - test_pre_pos, dim=-1)
                test_loss = torch.mean(test_diff)
                # print(f"test loss: {test_loss}")
                writer.add_scalar("rate/test_loss", test_loss.item(), epoch)

        bar.update(1)


torch.save(model.state_dict(), result_path)
