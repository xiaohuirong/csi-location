`{r}`对应比赛轮数，`{s}`代表场景， `{m}`代表使用的方法，`{dir}`定义为`data/round{r}/s{s}`。

初始数据组织结果如下:

```shell
tree.txt
└── data
    └── round{r}
        └── s{s}
            ├── data
            │   ├── Round{r}CfgData{s}.txt     # 配置文件
            │   ├── Round{r}GroundTruth{s}.txt # 所有点坐标
            │   ├── Round{r}InputData{s}.txt   # 所有信道数据
            │   └── Round{r}InputPos{s}.txt    # 锚点坐标
            ├── feature						   # 存放特征
            └── result						   # 存放数量网络参数和输出结果
```

### 方法一: 基于每个点的信道特征

#### 1. 数据类型转换

将txt文本格式的csi信道信息转化为numpy的.npy数据，方便后续计算操作

```shell
python 1_data_convert.py --round {r} --scene {s}
```

**输入:** 

- 配置文件:`{dir}/data/Round{r}CfgData{s}.txt`
- 信道数据:`{dir}/data/Round{r}InputData{s}.txt`

**输出:**

- npy格式信道数据:`{dir}/data/Round{r}InputData{s}.npy`

#### 2. 数据切片

切片出描点对应的信道信息

```shell
python 2_data_slice.py --round {r} --scene {s}
```

**输入:**

- npy格式信道数据:`{dir}/data/Round{r}InputData{s}.npy`
- 锚点坐标和索引:`{dir}/data/Round{r}InputPos{s}.txt`

**输出:**

- 锚点信道数据切片: `{dir}/data/Round{r}InputData{s}_S.npy`
- 锚点位置: `{dir}/data/Round{r}InputPos{s}_S.npy`
- 描点数据索引: `{dir}/data/Round{r}Index{s}_S.npy`

#### 3.  生成测试数据切片(可选)

切片2000组信道信息用于性能测试，实际比赛时无法生成测试数据

```shell
python 3_generate_test_data.py --round {r} --scene {s}
```

**输入:**

- npy格式信道数据:`{dir}/data/Round{r}InputData{s}.npy`
- 所有点坐标: `{dri}/data/Round{r}GroundTruth{s}.txt`

**输出:**

- 测试信道数据切片: `{dir}/data/TestRound{r}InputData{s}_S.npy`

- 测试信道对应的位置: `{dir}/data/TestRound{r}InputPos{s}_S.npy`

- 测试信道索引: `{dir}/data/Test{args.seed}Round{r}Index{s}_S.npy`

#### 4. 特征提取

1. 提取所有点信道特征

   ```shell
   python 4_feature_extract.py --round {r} --scene {s} --data Round{r}InputData{s}.npy --method {m}
   ```

   **输入:**

   - 所有点信道数据:`{dir}/data/Round{r}InputData{s}.npy`

   **输出:**

   - 所有信道特征:`{dir}/feature/{m}:FRound{r}InputData{s}.npy`

2. 提取锚点信道特征

   ```shell
   python 4_feature_extract.py --round {r} --scene {s} --data Round{r}InputData{s}_S.npy --method {m}
   ```

   **输入:**

   - 锚点信道数据:`{dir}/data/Round{r}InputData{s}_S.npy`

   **输出:**

   - 锚点信道特征:`{dir}/feature/{m}:FRound{r}InputData{s}_S.npy`

3. 提取测试信道特征(可选)

   ```shell
   python 4_feature_extract.py --round {r} --scene {s} --data TestRound{r}InputData{s}_S.npy --method {m}
   ```

   **输入:**

   - 测试信道数据:`{dir}/data/TestRound{r}InputData{s}_S.npy`

   **输出:**

   - 测试信道特征:`{dir}/feature/{m}:FTestRound{r}InputData{s}_S.npy`

#### 5. 训练

利用已有特征训练神经网络，并保存神经网络参数

```shell
python 5_train.py --round {r} --scene {s} --bsz 10 --method {m} --tseed 0 [--test]
```

可以指定`--test`参数来控制添加测试集测试

**输入：**

- 锚点特征: `{dir}/feature/{m}:FRound{r}InputData{s}_S.npy`
-  锚点位置: `{dir}/feature/Round{r}InputPos{s}_S.npy}`
- 测试点信道特征(可选): `{dir}/feature/{m}:FTestRound{r}InputData{s}_S.npy`
- 测试点位置(可选):`{dir}/feature/TestRound{r}InputPos{s}_S.npy}`

**输出:**

- 神经网络权重：`{dir}/result/{m}:M{args.tseed}Round{r}Scene{s}.pth`

#### 6. 估计位置

使用神经网络权重估计位置

```shell
python 6_cal_pos.py --round {r} --scene {s} --method {m}
```

**输入:**

- 神经网络权重：`{dir}/result/{m}:M{args.tseed}Round{r}Scene{s}.pth`
- 所有信道特征: `{dir}/feature/{m}:FRound{r}InputData{s}.npy`

**输出:**

- 所有点估计坐标: `{dir}/result/{m}:Round{r}OutputPos{s}.txt`



#### 7. 评估性能指标

评估具体场景下的性能差异

```shell
python 7_benchmark.py --round {m} --scene {m} --method {m}
```

**输入:**

- 所有点估计坐标: `{dir}/result/{m}:Round{r}OutputPos{s}.txt`
- 所有点真实坐标: `{dir}/data/Round{r}InputPos{s}.txt`

**输出:**

- 打印平均距离误差
- 在对应位置绘制估计点散点图