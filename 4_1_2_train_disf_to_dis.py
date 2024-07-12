import torch
from torch.utils.data import DataLoader
import numpy as np
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from utils.parse_args import parse_args, show_args
import tqdm
from net.MLP import (
    CustomDataset5,
    MLP2,
    set_seed,
)

args = parse_args()
set_seed(args.tseed)

r = args.round
s = args.scene

dir = f"data/round{r}/s{s}/data/"
feature_dir = f"data/round{r}/s{s}/feature/"

feature_path = feature_dir + f"FDis{r}Scene{s}_S.npy"
pos_path = dir + f"Round{r}InputPos{s}_S.npy"

test_feature_path = feature_dir + f"Test{args.seed}FDis{r}Scene{s}_S.npy"
test_pos_path = dir + f"Test{args.seed}Round{r}InputPos{s}_S.npy"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

feature = np.load(feature_path)
feature = torch.from_numpy(feature).float().to(device)
pos = np.load(pos_path)
pos = torch.from_numpy(pos).float().to(device)

feature = feature.view(2000 * 2000, 12)
dis = torch.norm(pos.view(1, 2000, 2) - pos.view(2000, 1, 2), dim=-1)
dis = dis.view(2000 * 2000, 1)
dataset = CustomDataset5(feature, dis)

if args.test:
    test_feature = np.load(test_feature_path)
    test_feature = torch.from_numpy(test_feature).float().to(device)
    test_pos = np.load(test_pos_path)
    test_pos = torch.from_numpy(test_pos).float().to(device)
    test_feature = test_feature.view(2000 * 2000, 12)
    test_dis = torch.norm(pos.view(1, 2000, 2) - pos.view(2000, 1, 2), dim=-1)
    test_dis = test_dis.view(2000 * 2000, 1)
    test_dataset = CustomDataset5(test_feature, test_dis)

# default : 2000 * 2000
batch_size = args.bsz

# input_dim : 12
input_dim = feature.shape[1]

# defautl : 1024
embedding_dim = args.embedding
if embedding_dim > 12:
    print(f"Warning: embedding_dim {embedding_dim} is too large.")

# defautl 1e-3
lr = args.lr
# default 200
step = args.step
# default 0.9
gamma = args.gamma
model = MLP2(input_dim, embedding_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=step, gamma=gamma)

writer = SummaryWriter(f"runs/DisR{r}S{s}-{args.time}")
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
            dis = batch["dis"]

            pre_dis = model(feature)

            diff = torch.abs(dis - pre_dis)
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
                test_feature = test_batch["feature"]
                test_dis = test_batch["dis"]
                test_pre_dis = model(test_feature)
                test_diff = torch.abs(test_dis - test_pre_dis)
                test_loss = torch.mean(test_diff)
                writer.add_scalar("rate/test_loss", test_loss.item(), epoch)
                break

        bar.update(1)

result_dir = f"data/round{r}/s{s}/result/"
result_path = result_dir + f"DisM{args.tseed}Round{r}Scene{s}.pth"

torch.save(model.state_dict(), result_path)
