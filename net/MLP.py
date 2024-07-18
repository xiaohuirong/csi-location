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


class CustomDataset6:
    def __init__(
        self, feature, dmg, known_pos, known_pos_index, in_index, out_index, device
    ):
        self.device = device
        self.feature = feature
        self.dmg = dmg
        self.known_pos = known_pos
        self.known_pos_index = known_pos_index
        self.in_index = in_index
        self.out_index = out_index

        self.known_feature = self.feature[self.known_pos_index]
        self.out_feature = self.feature[self.out_index]
        self.known_num = self.known_feature.shape[0]
        self.out_num = self.out_feature.shape[0]

        self.known_dis = torch.norm(
            self.known_pos.view(self.known_num, 1, 2)
            - self.known_pos.view(1, self.known_num, 2),
            dim=-1,
        )
        self.known_dmg = self.dmg[self.known_pos_index, :][:, self.known_pos_index]
        self.known_out_dmg = self.dmg[self.known_pos_index, :][:, self.out_index]
        self.out_dmg = self.dmg[self.out_index, :][:, self.out_index]

    def get_known_feature_pos(self, mini_bsz, shuffle=False):
        if shuffle:
            idxs = torch.randperm(self.known_num, device=self.device)
        else:
            idxs = torch.arange(self.known_num, device=self.device)

        num_batches = (self.known_num + mini_bsz - 1) // mini_bsz

        mini_batches = []
        for i in range(num_batches):
            start_idx = i * mini_bsz
            end_idx = min(start_idx + mini_bsz, self.known_num)
            idx = idxs[start_idx:end_idx]
            mini_batch = {
                "feature": self.known_feature[idx],
                "pos": self.known_pos[idx],
            }
            mini_batches.append(mini_batch)

        return mini_batches

    def get_known_pos_dmg(self, mini_bsz, shuffle=False):
        if shuffle:
            idxs = torch.randperm(self.known_num, device=self.device)
        else:
            idxs = torch.arange(self.known_num, device=self.device)

        num_batches = (self.known_num + mini_bsz - 1) // mini_bsz

        mini_batches = []
        for i in range(num_batches):
            start_idx = i * mini_bsz
            end_idx = min(start_idx + mini_bsz, self.known_num)
            idx = idxs[start_idx:end_idx]
            mini_batch = {
                "pos": self.known_pos[idx],
                "dmg": self.known_dmg[idx, :][:, idx],
                "index": idx,
            }
            mini_batches.append(mini_batch)

        return mini_batches

    def get_known_out(self, mini_bsz, shuffle=False):
        if shuffle:
            idxs = torch.randperm(self.out_num, device=self.device)
        else:
            idxs = torch.arange(self.out_num, device=self.device)

        num_batches = (self.out_num + mini_bsz - 1) // mini_bsz

        mini_batches = []
        for i in range(num_batches):
            start_idx = i * mini_bsz
            end_idx = min(start_idx + mini_bsz, self.out_num)
            idx = idxs[start_idx:end_idx]

            mini_batch = {
                "feature": self.out_feature[idx],
                "pos": self.known_pos,
                "dmg": self.known_out_dmg[:, idx],
            }
            mini_batches.append(mini_batch)

        return mini_batches

    def get_out(self, mini_bsz, shuffle=False):
        if shuffle:
            idxs = torch.randperm(self.out_num, device=self.device)
        else:
            idxs = torch.arange(self.out_num, device=self.device)

        num_batches = (self.out_num + mini_bsz - 1) // mini_bsz

        mini_batches = []
        for i in range(num_batches):
            start_idx = i * mini_bsz
            end_idx = min(start_idx + mini_bsz, self.out_num)
            idx = idxs[start_idx:end_idx]

            mini_batch = {
                "feature": self.out_feature[idx],
                "dmg": self.out_dmg[idx, :][:, idx],
            }
            mini_batches.append(mini_batch)

        return mini_batches

    def get_in_data(self):
        pass

    def get_out_data(self):
        pass


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


class SMLP(nn.Module):
    def __init__(self):
        super(SMLP, self).__init__()

        self.fc1 = nn.Linear(2, 2)

    def forward(self, x):
        x = self.fc1(x)
        return x


def cal_loss(pre_pos1, pre_pos2, h_dis, threshold=50):
    """
    input:
        pre_pos: (bsz, 2) -> pre_dis: (bsz, bsz)
        h_dis: (bsz, bsz)

    output:
        loss: pre_dis - h_dis
    """
    bsz1 = pre_pos1.shape[0]
    bsz2 = pre_pos2.shape[0]

    mask = h_dis > threshold

    # diff_pos (bsz, bsz, 2)
    diff_pos = pre_pos1.reshape(bsz1, 1, 2) - pre_pos2.reshape(1, bsz2, 2)

    # pre_dis (bsz, bsz)
    pre_dis = torch.norm(diff_pos, dim=-1)

    losses = torch.abs(pre_dis - h_dis)
    losses[mask] = 0

    loss = torch.mean(losses)

    return loss


if __name__ == "__main__":
    # 使用自定义 DataLoader
    dataset = ExampleDataset([i for i in range(100)])
    data = np.random.rand(100, 100)
    custom_dataloader = CustomDataLoader(dataset, batch_size=5, data=data)

    for batch in custom_dataloader:
        print(batch)
