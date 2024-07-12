import torch
import random
import numpy as np
from torch.utils.data import DataLoader, Dataset, Sampler
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


class CustomBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, data):
        self.dataset = dataset
        self.batch_size = batch_size
        self.l_dataset = len(dataset)
        self.indices = list(range(self.l_dataset))
        self.data = data
        self.batch_num = self.l_dataset // self.batch_size

        self.sampled_list = random.sample(self.indices, self.batch_num)

    def __iter__(self):
        # 实现自定义的分批规则，这里简单实现按顺序分批
        batches = []

        for i in self.sampled_list:
            _, batch = torch.topk(self.data[i], self.batch_size, largest=False)
            batches.append(batch)
        return iter(batches)

    def __len__(self):
        return len(self.indices) // self.batch_size


class CustomDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, data=None, **kwargs):
        batch_sampler = CustomBatchSampler(dataset, batch_size, data)
        super().__init__(dataset, batch_sampler=batch_sampler, **kwargs)

    def __iter__(self):
        for indices in self.batch_sampler:
            batch = self.dataset[indices]
            yield batch


# 示例数据集
class ExampleDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class CustomDataset(Dataset):
    def __init__(self, feature, pos):
        self.feature = feature
        self.pos = pos

    def __len__(self):
        return len(self.feature)

    def __getitem__(self, idx):
        sample = {
            "feature": self.feature[idx],
            "pos": self.pos[idx % 2000],
            "index": idx,
        }
        return sample


class CustomDataset3(Dataset):
    def __init__(self, feature, pos):
        bsz = feature.shape[0]
        self.feature = feature
        self.pos = pos

        # self.feature : (bsz, bsz, 1)
        self.feature = self.feature.reshape(bsz, bsz, 1)
        # self.pos_r : (1, bsz, 2)
        self.pos_r = self.pos.reshape(1, bsz, 2)
        # self.pos_r : (bsz, bsz, 2)
        self.pos_r = self.pos_r.repeat(bsz, 1, 1)

        # self.feature : (bsz, bsz, 3)
        self.feature = torch.cat((self.feature, self.pos_r), axis=-1)

    def __len__(self):
        return len(self.feature)

    def __getitem__(self, idx):
        sample = {
            "feature": self.feature[idx],
            "pos": self.pos[idx],
            "index": idx,
        }
        return sample


class CustomDataset5(Dataset):
    def __init__(self, feature, dis):
        self.feature = feature
        self.dis = dis

    def __len__(self):
        return len(self.feature)

    def __getitem__(self, idx):
        sample = {
            "feature": self.feature[idx],
            "dis": self.dis[idx],
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


class MLP2(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(MLP2, self).__init__()

        # 定义嵌入层
        self.embedding = nn.Linear(input_dim, embedding_dim)

        # 定义全连接层
        self.fc1 = nn.Linear(embedding_dim, 6)
        self.fc2 = nn.Linear(6, 1)

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

        return x


def cal_loss(pre_pos, h_dis):
    """
    input:
        pre_pos: (bsz, 2) -> pre_dis: (bsz, bsz)
        h_dis: (bsz, bsz)

    output:
        loss: pre_dis - h_dis
    """
    bsz = pre_pos.shape[0]

    # diff_pos (bsz, bsz, 2)
    diff_pos = pre_pos.reshape(bsz, 1, 2) - pre_pos.reshape(1, bsz, 2)

    # pre_dis (bsz, bsz)
    pre_dis = torch.norm(diff_pos, dim=-1)

    loss = torch.mean(torch.abs(pre_dis - h_dis))

    return loss


if __name__ == "__main__":
    # 使用自定义 DataLoader
    dataset = ExampleDataset([i for i in range(100)])
    data = np.random.rand(100, 100)
    custom_dataloader = CustomDataLoader(dataset, batch_size=5, data=data)

    for batch in custom_dataloader:
        print(batch)
