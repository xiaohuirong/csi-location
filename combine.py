import numpy as np
from utils.parse_args import parse_args, show_args

args = parse_args()
show_args(args)
r = args.round
m = args.method


poses = None
features = None
test_poses = None
test_features = None
for s in range(3):
    data_dir = f"data/round{r}/s{s+1}/data/"
    feature_dir = f"data/round{r}/s{s+1}/feature/"

    pos = np.load(data_dir + f"Round{r}InputPos{s+1}_S.npy")
    feature = np.load(feature_dir + f"{m}:" + "F" + f"Round{r}InputData{s+1}_S.npy")
    data = np.load(data_dir + f"Round{r}InputData{s+1}_S.npy").astype(np.complex64)

    test_pos = np.load(data_dir + f"Test{args.seed}Round{r}InputPos{s+1}_S.npy")
    test_feature = np.load(
        feature_dir + f"{m}:" + "F" + f"Test{args.seed}Round{r}InputData{s+1}_S.npy"
    )
    test_data = np.load(
        data_dir + f"Test{args.seed}Round{r}InputData{s+1}_S.npy"
    ).astype(np.complex64)

    if poses is None:
        poses = pos
        features = feature
        datas = data
        test_poses = test_pos
        test_features = test_feature
        test_datas = test_data
    else:
        poses = np.concatenate((poses, pos), axis=0)
        features = np.concatenate((features, feature), axis=0)
        datas = np.concatenate((datas, data), axis=0)
        test_poses = np.concatenate((test_poses, test_pos), axis=0)
        test_features = np.concatenate((test_features, test_feature), axis=0)
        test_datas = np.concatenate((test_datas, test_data), axis=0)

print(poses.shape)
print(test_poses.shape)
print(features.shape)
print(test_features.shape)


s_3 = np.sqrt(3)
A = [[200, 0], [0, 200]]
if r == 0:
    P = [[-100 * s_3, -100], [100 * s_3, -100], [0, 200]]
elif r == 1:
    P = [[100 * s_3, -100], [0, 200], [-100 * s_3, -100]]

Ts = []
t_poses = None
t_test_poses = None
k = 0
for i in range(3):
    p1 = P[i % 3]
    p2 = P[(i + 1) % 3]
    B = np.zeros((2, 2))
    B[:, 0] = p1
    B[:, 1] = p2
    print(B)
    B = np.linalg.inv(B)
    T = np.dot(A, B)
    Ts.append(T)
    if t_poses is None:
        t_poses = np.dot(T, poses[2000 * k : 2000 * (k + 1)].T).T
        t_test_poses = np.dot(T, test_poses[2000 * k : 2000 * (k + 1)].T).T
    else:
        t_poses = np.concatenate(
            (t_poses, np.dot(T, poses[2000 * k : 2000 * (k + 1)].T).T), axis=-1
        )
        t_test_poses = np.concatenate(
            (t_test_poses, np.dot(T, test_poses[2000 * k : 2000 * (k + 1)].T).T),
            axis=-1,
        )
    k += 1


import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(t_test_poses[:, 0], t_test_poses[:, 1])
plt.show()

s = 4

data_dir = f"data/round{r}/s{s}/data/"
feature_dir = f"data/round{r}/s{s}/feature/"
np.save(data_dir + f"Round{r}InputPos{s}_S.npy", t_poses)
np.save(feature_dir + f"{m}:" + "F" + f"Round{r}InputData{s}_S.npy", features)
np.save(data_dir + f"Round{r}InputData{s}_S.npy", datas.astype(np.complex64))

np.save(data_dir + f"Test{args.seed}Round{r}InputPos{s}_S.npy", t_test_poses)
np.save(
    feature_dir + f"{m}:" + "F" + f"Test{args.seed}Round{r}InputData{s}_S.npy",
    test_features,
)
np.save(
    data_dir + f"Test{args.seed}Round{r}InputData{s}_S.npy",
    test_datas.astype(np.complex64),
)
