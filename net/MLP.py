import torch
import random
import numpy as np
from torch.utils.data import Dataset
import os
import torch.nn as nn

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


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


class MLP(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(MLP, self).__init__()

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
