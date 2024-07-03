import torch
import numpy as np
import torch.nn as nn
from utils.parse_args import parse_args, show_args
import random


def set_seed(tseed):
    torch.manual_seed(tseed)
    torch.cuda.manual_seed(tseed)
    torch.cuda.manual_seed_all(tseed)  # if you are using multi-GPU.
    np.random.seed(tseed)
    random.seed(tseed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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

result_dir = f"data/round{r}/s{s}/result/"
weight_path = result_dir + f"M{args.tseed}Round{r}Scene{s}.pth"

result_path = result_dir + f"Round{r}OutputPos{s}.txt"

test_feature_path = feature_dir + f"FRound{r}InputData{s}.npy"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_feature = np.load(test_feature_path)
test_feature = torch.from_numpy(test_feature).float().to(device)

# input_dim = 2 * 2 * 8 * 4 * 2 + 408
input_dim = test_feature.shape[1]

# defautl : 1024
embedding_dim = args.embedding

model = CustomNetwork(input_dim, embedding_dim).to(device)
model.load_state_dict(torch.load(weight_path))

show_args(args)

test_pre_pos = model(test_feature)
np.savetxt(result_path, test_pre_pos.detach().cpu().numpy(), fmt="%.4f")
