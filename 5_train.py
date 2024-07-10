import torch
from torch.utils.data import DataLoader
import numpy as np
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from utils.parse_args import parse_args, show_args
import tqdm
from net.MLP import CustomDataset, MLP, set_seed, CustomDataset3

args = parse_args()
set_seed(args.tseed)

r = args.round
s = args.scene
m = args.method

dir = f"data/round{r}/s{s}/data/"
feature_dir = f"data/round{r}/s{s}/feature/"

feature_path = feature_dir + f"{m}:FRound{r}InputData{s}_S.npy"
pos_path = dir + f"Round{r}InputPos{s}_S.npy"

test_feature_path = feature_dir + f"{m}:FTest{args.seed}Round{r}InputData{s}_S.npy"
test_pos_path = dir + f"Test{args.seed}Round{r}InputPos{s}_S.npy"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


feature = np.load(feature_path)
pos = np.load(pos_path)
feature = torch.from_numpy(feature).float().to(device)
pos = torch.from_numpy(pos).float().to(device)

if args.test:
    test_feature = np.load(test_feature_path)
    test_pos = np.load(test_pos_path)
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

p = 100

# defautl 1e-3
lr = args.lr
# default 200
step = args.step
# default 0.9
gamma = args.gamma
if m == 3:
    model = MLP(input_dim // p * 3, embedding_dim).to(device)
else:
    model = MLP(input_dim, embedding_dim).to(device)
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

if args.test:
    test_dataloader = DataLoader(test_dataset, batch_size=2000, shuffle=False)

with tqdm.tqdm(total=epoch_num) as bar:
    for epoch in range(epoch_num):
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        all_loss = []
        for batch in dataloader:
            if m == 3:
                feature = batch["feature"].reshape(batch_size * p, input_dim // p * 3)
                pos = batch["pos"].repeat_interleave(p, dim=0)
            else:
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

        if epoch % 100 == 0 and args.test:
            for test_batch in test_dataloader:
                if m == 3:
                    test_feature = test_batch["feature"].reshape(
                        2000 * p, input_dim // p * 3
                    )
                    test_pos = test_batch["pos"].repeat_interleave(p, dim=0)
                else:
                    test_feature = test_batch["feature"]
                    test_pos = test_batch["pos"]
                test_pre_pos = model(test_feature)
                test_diff = torch.norm(test_pos - test_pre_pos, dim=-1)
                test_loss = torch.mean(test_diff)
                # print(f"test loss: {test_loss}")
                writer.add_scalar("rate/test_loss", test_loss.item(), epoch)

        bar.update(1)

result_dir = f"data/round{r}/s{s}/result/"
result_path = result_dir + f"{m}:M{args.tseed}Round{r}Scene{s}.pth"

torch.save(model.state_dict(), result_path)
