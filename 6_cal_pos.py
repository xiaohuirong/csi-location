import torch
import numpy as np
from utils.parse_args import parse_args, show_args
from net.MLP import MLP, set_seed
from utils.cal_utils import rotate_center_back

args = parse_args()
show_args(args)
set_seed(args.tseed)

r = args.round
s = args.scene

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

feature = np.load(args.feature_path)
feature = torch.from_numpy(feature).float().to(device)

# input_dim = 2 * 2 * 8 * 4 * 2 + 408
input_dim = feature.shape[1]

# defautl : 1024
embedding_dim = args.embedding

model = MLP(input_dim, embedding_dim).to(device)
model.load_state_dict(torch.load(args.pth_path))

test_pre_pos = model(feature)
test_pre_pos = test_pre_pos.detach().cpu().numpy()
test_pre_pos = rotate_center_back(test_pre_pos, s, r)

np.save(args.output_pos_path, test_pre_pos)
