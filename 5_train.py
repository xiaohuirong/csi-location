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
)
from utils.cal_utils import rotate_center_to_y

args = parse_args()
show_args(args)

set_seed(args.tseed)

r = args.round
s = args.scene

feature_path = args.feature_slice_path
pos_path = args.pos_path

test_feature_path = args.test_feature_slice_path
test_pos_path = args.test_pos_path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pos = np.load(pos_path)
pos = rotate_center_to_y(pos, s, r)
feature = np.load(feature_path)

if r == 2 and s % 3 == 0:
    s2 = s - 2
    dir2 = f"data/round{r}/s{s2}/data/"
    pos_path2 = dir2 + f"Round{r}InputPos{s2}_S.npy"
    pos2 = np.load(pos_path2)
    pos2 = rotate_center_to_y(pos2, s2, r)
    pos = np.vstack((pos, pos2))

    feature_dir2 = f"data/round{r}/s{s2}/feature/"
    feature_path2 = feature_dir2 + f"FRound{r}InputData{s2}_S.npy"
    feature2 = np.load(feature_path2)
    feature = np.vstack((feature, feature2))

feature = torch.from_numpy(feature).float().to(device)
pos = torch.from_numpy(pos).float().to(device)

if args.test:
    test_feature = np.load(test_feature_path)
    test_pos = np.load(test_pos_path)
    test_pos = rotate_center_to_y(test_pos, s, r)
    test_feature = torch.from_numpy(test_feature).float().to(device)
    test_pos = torch.from_numpy(test_pos).float().to(device)

dataset = CustomDataset(feature, pos)
if args.test:
    test_dataset = CustomDataset(test_feature, test_pos)

# default : 100
batch_size = args.bsz

# input_dim = 2 * 2 * 8 * 4 * 2 + 408
input_dim = feature.shape[1]

# defautl : 1024
embedding_dim = args.embedding

# defautl 1e-3
lr = args.lr
# default 200
step = args.step
# default 0.9
gamma = args.gamma
model = MLP(input_dim, embedding_dim).to(device)

if args.cp != "None":
    model.load_state_dict(torch.load(args.cp))

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
scheduler = lr_scheduler.StepLR(optimizer, step_size=step, gamma=gamma)

writer = SummaryWriter(f"runs/R{r}S{s}-{args.time}")
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

with tqdm.tqdm(total=epoch_num) as bar:
    for epoch in range(epoch_num):
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        all_loss = []
        for batch in dataloader:
            feature = batch["feature"]
            pos = batch["pos"]
            index = batch["index"]

            pre_pos = model(feature)

            diff = torch.norm(pos - pre_pos, dim=-1)
            loss = torch.mean(diff)

            all_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        mean_all_loss = torch.mean(torch.tensor(all_loss))
        writer.add_scalar("rate/train_loss", mean_all_loss.item(), epoch)

        scheduler.step()

        if epoch % 100 == 0 and args.test:
            for test_batch in test_dataloader:
                test_feature = test_batch["feature"]
                test_pos = test_batch["pos"]
                test_pre_pos = model(test_feature)
                test_diff = torch.norm(test_pos - test_pre_pos, dim=-1)
                test_loss = torch.mean(test_diff)
                writer.add_scalar("rate/test_loss", test_loss.item(), epoch)

        bar.update(1)

torch.save(model.state_dict(), args.pth_path)
