import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

from torch.optim import lr_scheduler

import torch.nn as nn
import torch.optim as optim

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from torch.utils.tensorboard import SummaryWriter

from utils.parse_args import parse_args, show_args

import random

import tqdm


def set_seed(tseed):
    torch.manual_seed(tseed)
    torch.cuda.manual_seed(tseed)
    torch.cuda.manual_seed_all(tseed)  # if you are using multi-GPU.
    np.random.seed(tseed)
    random.seed(tseed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CustomDataset(Dataset):
    def __init__(self, feature, pos):
        self.feature = feature
        self.pos = pos

    def __len__(self):
        return len(self.feature)

    def __getitem__(self, idx):
        sample = {
            "feature": self.feature[idx],
            "pos": self.pos[idx],
        }
        return sample


class CustomNetwork(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(CustomNetwork, self).__init__()

        # 定义嵌入层
        self.embedding = nn.Linear(input_dim, embedding_dim)

        # 定义全连接层
        self.fc1 = nn.Linear(embedding_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 2)

        # 定义激活函数
        self.relu = nn.ReLU()

    def forward(self, x):
        # 前向传播
        x = self.embedding(x)
        x = self.relu(x)

        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.relu(x)

        x = self.fc3(x)
        x = self.relu(x)

        x = self.fc4(x)
        x = self.relu(x)

        x = self.fc5(x)

        return x


args = parse_args()
set_seed(args.tseed)

r = args.round
s = args.scene

dir = f"data/round{r}/s{s}/data/"
feature_dir = f"data/round{r}/s{s}/feature/"

feature_path = feature_dir + f"F2Round{r}InputData{s}_S.npy"
pos_path = dir + f"Round{r}InputPos{s}_S.npy"

test_feature_path = feature_dir + f"F2Test{args.seed}Round{r}InputData{s}_S.npy"
test_pos_path = dir + f"Test{args.seed}Round{r}InputPos{s}_S.npy"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


feature = np.load(feature_path)
pos = np.load(pos_path)
feature = torch.from_numpy(feature).float().to(device)
pos = torch.from_numpy(pos).float().to(device)
dataset = CustomDataset(feature, pos)

if args.test:
    test_feature = np.load(test_feature_path)
    test_pos = np.load(test_pos_path)
    test_feature = torch.from_numpy(test_feature).float().to(device)
    test_pos = torch.from_numpy(test_pos).float().to(device)

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
model = CustomNetwork(input_dim, embedding_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=step, gamma=gamma)

writer = SummaryWriter(f"runs/Round{r}Scene{s}-{args.time}")
writer.add_text(
    "parameter",
    "|param|value|\n|-|-|\n%s"
    % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
)

show_args(args)
# default : 5000
epoch_num = args.epoch

with tqdm.tqdm(total=epoch_num) as bar:
    for epoch in range(epoch_num):
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        all_loss = []
        for batch in dataloader:
            feature = batch["feature"]
            pos = batch["pos"]

            pre_pos = model(feature)

            diff = torch.norm(pos - pre_pos, dim=-1)

            loss = torch.mean(diff)

            all_loss.append(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        mean_all_loss = torch.mean(torch.tensor(all_loss))
        # print(f"train loss: {mean_all_loss.item()}")
        writer.add_scalar("rate/train_loss", mean_all_loss.item(), epoch)

        scheduler.step()

        if epoch % 100 == 0:
            if args.test:
                test_pre_pos = model(test_feature)
                test_diff = torch.norm(test_pos - test_pre_pos, dim=-1)
                test_loss = torch.mean(test_diff)
                # print(f"test loss: {test_loss}")
                writer.add_scalar("rate/test_loss", test_loss.item(), epoch)

        bar.update(1)

result_dir = f"data/round{r}/s{s}/result/"
result_path = result_dir + f"2M{args.tseed}Round{r}Scene{s}.pth"

torch.save(model.state_dict(), result_path)
