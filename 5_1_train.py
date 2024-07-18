import torch
from torch.utils.data import DataLoader
import numpy as np
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from utils.parse_args import parse_args, show_args
import tqdm
from utils.plot_utils import DA, two_plot, pos_plot
import torch.nn as nn
from net.MLP import (
    CustomDataset6,
    MLP,
    set_seed,
    cal_loss,
)

args = parse_args()
set_seed(args.tseed)

r = args.round
s = args.scene
m = args.method
p = args.port
o = args.over

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_dir = f"data/round{r}/s{s}/data/"
feature_dir = f"data/round{r}/s{s}/feature/"

out_index_path = data_dir + f"ClusterRound{r}Index{s}_S.npy"
out_index = np.load(out_index_path)
print(out_index.shape)

in_index = np.arange(20000)
in_index = in_index[~np.isin(in_index, out_index)]
print(in_index.shape)

known_pos_path = data_dir + f"Round{r}InputPos{s}_S.npy"
known_pos = np.load(known_pos_path)

known_pos_index_path = data_dir + f"Round{r}Index{s}_S.npy"
known_pos_index = np.load(known_pos_index_path)

feature_path = feature_dir + f"{m}:FRound{r}InputData{s}.npy"
feature = np.load(feature_path)

dmg_path = feature_dir + f"Port{p}Over{o}Dmg{r}Scene{s}.npy"
dmg = np.load(dmg_path)

# da = DA(dmg, known_pos_index, in_index, out_index)
# da.plot()
# exit()

known_pos = torch.from_numpy(known_pos).float().to(device)
feature = torch.from_numpy(feature).float().to(device)
dmg = torch.from_numpy(dmg).to(device)

dataset = CustomDataset6(
    feature, dmg, known_pos, known_pos_index, in_index, out_index, device=device
)

# import time
# time.sleep(10000)
# exit()

# default : 100
batch_size = args.bsz

# input_dim = 2 * 2 * 8 * 4 * 2 + 408
input_dim = feature.shape[1]
# input_dim = 2000

# defautl : 1024
embedding_dim = args.embedding

# defautl 1e-3
lr = args.lr
# default 200
step = args.step
# default 0.9
gamma = args.gamma

pos_t = nn.Linear(2, 2, bias=False, device=device)
optimizer_pos_t = optim.Adam(pos_t.parameters(), lr=lr)
scheduler_pos_t = lr_scheduler.StepLR(optimizer_pos_t, step_size=step, gamma=gamma)

model = MLP(input_dim, embedding_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=step, gamma=gamma)

model2 = MLP(input_dim, embedding_dim).to(device)
optimizer2 = optim.Adam(model2.parameters(), lr=lr)
scheduler2 = lr_scheduler.StepLR(optimizer2, step_size=step, gamma=gamma)


model3 = MLP(input_dim, embedding_dim).to(device)
optimizer3 = optim.Adam(model3.parameters(), lr=lr)
scheduler3 = lr_scheduler.StepLR(optimizer3, step_size=step, gamma=gamma)

models = []
models.append(pos_t)
models.append(model)
models.append(model2)
models.append(model3)

optimizers = []
optimizers.append(optimizer_pos_t)
optimizers.append(optimizer)
optimizers.append(optimizer2)
optimizers.append(optimizer3)

schedulers = []
schedulers.append(scheduler_pos_t)
schedulers.append(scheduler)
schedulers.append(scheduler2)
schedulers.append(scheduler3)

writer = SummaryWriter(f"runs/R{r}S{s}M{m}-{args.time}")
writer.add_text(
    "parameter",
    "|param|value|\n|-|-|\n%s"
    % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
)

show_args(args)
# default : 5000
epoch_num = args.epoch

for i in range(3):
    model = models[i]
    optimizer = optimizers[i]
    scheduler = schedulers[i]
    if i == 2:
        model.load_state_dict(models[1].state_dict())
    if i == 3:
        model.load_state_dict(models[2].state_dict())

    with tqdm.tqdm(total=epoch_num) as bar:
        for epoch in range(epoch_num):
            if i == 0:
                dataloader = dataset.get_known_pos_dmg(
                    mini_bsz=batch_size, shuffle=False
                )
            elif i == 1:
                dataloader = dataset.get_known_feature_pos(
                    mini_bsz=batch_size, shuffle=False
                )
            elif i == 2:
                dataloader = dataset.get_known_out(mini_bsz=batch_size, shuffle=False)
                dataloader2 = dataset.get_out(mini_bsz=batch_size, shuffle=False)

            elif i == 3:
                dataloader = dataset.get_out(mini_bsz=batch_size, shuffle=False)
            all_loss = []
            poses = []
            k = 0
            for batch in dataloader:
                if i == 0:
                    dmg = batch["dmg"]
                    pos = batch["pos"]
                    chart_pos = model(pos)
                    loss = cal_loss(chart_pos, chart_pos, dmg, 50)
                elif i == 1:
                    feature = batch["feature"]
                    pos = batch["pos"]
                    pre_pos = model(feature)
                    diff = torch.norm(pos - pre_pos, dim=-1)
                    loss = torch.mean(diff)
                elif i == 2:
                    feature = batch["feature"]
                    dmg = batch["dmg"]
                    pos = batch["pos"]

                    with torch.no_grad():
                        chart_pos = models[0](pos)
                    pre_pos = model(feature)
                    chart_pre_pos = models[0](pre_pos)

                    loss = cal_loss(chart_pos, chart_pre_pos, dmg, 50)

                    feature = dataloader2[k]["feature"]
                    dmg = dataloader2[k]["dmg"]
                    pre_pos = model(feature)
                    chart_pre_pos = models[0](pre_pos)
                    loss2 = cal_loss(chart_pre_pos, chart_pre_pos, dmg, 50)

                    loss = loss + loss2

                    # poses.append(pre_pos.detach().cpu().numpy())

                elif i == 3:
                    feature = batch["feature"]
                    dmg = batch["dmg"]

                    pre_pos = model(feature)
                    chart_pre_pos = models[0](pre_pos)

                    loss = cal_loss(chart_pre_pos, chart_pre_pos, dmg)

                    poses.append(pre_pos.detach().cpu().numpy())
                k += 1

                all_loss.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if i == 2 or i == 3:
                poses.append(pos.detach().cpu().numpy())

            mean_all_loss = torch.mean(torch.tensor(all_loss))
            writer.add_scalar(f"rate/train_loss{i}", mean_all_loss.item(), epoch)
            scheduler.step()
            bar.update(1)
        if i == 2 or i == 3:
            dataloader2 = dataset.get_out(mini_bsz=len(out_index), shuffle=False)
            feature = dataloader2[0]["feature"]
            pre_pos = model(feature)
            poses.append(pre_pos.detach().cpu().numpy())
            pos_plot(poses)
            np.save("data/pre_pose.npy", poses[-1])


result_dir = f"data/round{r}/s{s}/result/"
result_path = result_dir + f"{m}:M{args.tseed}Round{r}Scene{s}.pth"
torch.save(model.state_dict(), result_path)
