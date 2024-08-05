# 使用CSI信息对用户进行定位

**运行环境:**

- CPU: i5-13400
- GPU: RTX4060
- OS: ArchLinux 
- python: 3.12
- python依赖

```txt
Package                  Version
------------------------ -----------
matplotlib               3.9.1
numpy                    2.0.1
rich                     13.7.1
scipy                    1.14.0
tensorboard              2.17.0
torch                    2.4.0
tqdm                     4.66.4
```

## 0. 数据组织结构

`{r}`对应比赛轮数，`{s}`代表场景。

```shell
root
└── data
    └── round{r}
        └── s{s}
            ├── data
            │   ├── Round{r}CfgData{s}.txt     # 场景配置
            │   ├── Round{r}Index{s}_S.npy     # 锚点索引
            │   ├── Round{r}InputData{s}.npy   # CSI信息
            │   ├── Round{r}InputData{s}_S.npy # 锚点CSI信息
            │   ├── Round{r}InputPos{s}_S.npy  # 锚点位置
            │   └── Round{r}InputPos{s}.txt    # 锚点位置(txt)
            ├── feature
            │   ├── FRound{r}InputData{s}.npy  # 特征
            │   └── FRound{r}InputData{s}_S.npy# 锚点特征
            └── result
                ├── M0Round{r}Scene{s}.pth     # 模型参数
                ├── Round{r}OutputPos{s}.npy   # 输出坐标
                └── Round{r}OutputPos{s}.txt   # 处理后的输出坐标
```

## 1. 数据类型转换

将txt文本格式的csi信道信息转化为numpy的.npy数据，方便后续计算操作

```shell
python 1_data_convert.py --round {r} --scene {s}
```

## 2. 数据切片

切片出描点对应的信道信息

```shell
python 2_data_slice.py --round {r} --scene {s}
```

## 3.  生成测试数据切片(可选)

切片2000组信道信息用于性能测试，实际比赛时无法生成测试数据

```shell
python 3_generate_test_data.py --round {r} --scene {s}
```

## 4. 特征提取

主要考虑的特征为天线间的相位差、PDP曲线、AoA和ToF

1. 提取所有点信道特征

   ```shell
   python 4_feature_extract.py --round {r} --scene {s}
   ```

2. 提取锚点信道特征

   ```shell
   python 4_feature_extract.py --round {r} --scene {s} --slice
   ```

3. 提取测试信道特征(可选)

   ```shell
   python 4_feature_extract.py --round {r} --scene {s} --test
   ```


## 5. 训练

利用已有特征训练神经网络，并保存神经网络参数

```shell
python 5_train.py --round {r} --scene {s} [--bsz 10] [--tseed 0] [--test]
```

- `--bsz`: 指定批次大小
- `--tseed`:指定训练随机种子
- `--test`:是否使用测试数据检测模型精确度

## 6. 估计位置(推理的过程)

使用神经网络权重估计位置

```shell
python 6_cal_pos.py --round {r} --scene {s}
```

## 7. 修正位置

修正超出扇区范围的数据，使其处于扇区内

```shell
python 7_fix_out_of_range.py --round {r} --scene {s}
```

## 8. 数据回填

将锚点位置回填到预测的位置中

```shell
python 8_fill_back.py --round {r} --scene {s}
```



## 针对复赛数据的复现步骤：

针对推理过程，只需要运行`6_cal_pos.py`、`7_fix_out_of_range.py`、`8_fill_back.py`。

场景1

```shell
python 1_data_convert.py --round 2 --scene 1 # 需要把解压的txt原始信道数据放置到data/round2/s{s}/data中
python 2_data_slice.py --round 2 --scene 1
python 4_feature_extract.py --round 2 --scene 1
python 4_feature_extract.py --round 2 --scene 1 --slice
python 5_train.py --round 2 --scene 1 --bsz 10
python 6_cal_pos.py --round 2 --scene 1
python 7_fix_out_of_range.py --round 2 --scene 1
python 8_fill_back.py --round 2 --scene 1
```

场景2

```shell
python 1_data_convert.py --round 2 --scene 2 # 需要把解压的txt原始信道数据放置到data/round2/s{s}/data中
python 2_data_slice.py --round 2 --scene 2
python 4_feature_extract.py --round 2 --scene 2
python 4_feature_extract.py --round 2 --scene 2 --slice
python 5_train.py --round 2 --scene 2 --bsz 10
python 6_cal_pos.py --round 2 --scene 2
python 7_fix_out_of_range.py --round 2 --scene 2
python 8_fill_back.py --round 2 --scene 2
```

场景3

```shell
python 1_data_convert.py --round 2 --scene 3 # 需要把解压的txt原始信道数据放置到data/round2/s{s}/data中
python 2_data_slice.py --round 2 --scene 3
python 4_feature_extract.py --round 2 --scene 3
python 4_feature_extract.py --round 2 --scene 3 --slice
python 5_train.py --round 2 --scene 3 --bsz 10 --cp data/round2/s1/result/M0Round2Scene1.pth
python 6_cal_pos.py --round 2 --scene 3
python 7_fix_out_of_range.py --round 2 --scene 3
python 8_fill_back.py --round 2 --scene 3
```

场景4

```shell
python 1_data_convert.py --round 2 --scene 4 # 需要把解压的txt原始信道数据放置到data/round2/s{s}/data中
python 2_data_slice.py --round 2 --scene 4
python 4_feature_extract.py --round 2 --scene 4
python 4_feature_extract.py --round 2 --scene 4 --slice
python 5_train.py --round 2 --scene 4 --bsz 10
python 6_cal_pos.py --round 2 --scene 4
python 7_fix_out_of_range.py --round 2 --scene 4
python 8_fill_back.py --round 2 --scene 4
```

场景5

```shell
python 1_data_convert.py --round 2 --scene 5 # 需要把解压的txt原始信道数据放置到data/round2/s{s}/data中
python 2_data_slice.py --round 2 --scene 5
python 4_feature_extract.py --round 2 --scene 5
python 4_feature_extract.py --round 2 --scene 5 --slice
python 5_train.py --round 2 --scene 5 --bsz 10
python 6_cal_pos.py --round 2 --scene 5
python 7_fix_out_of_range.py --round 2 --scene 5
python 8_fill_back.py --round 2 --scene 5
```

场景6

```shell
python 1_data_convert.py --round 2 --scene 6 # 需要把解压的txt原始信道数据放置到data/round2/s{s}/data中
python 2_data_slice.py --round 2 --scene 6
python 4_feature_extract.py --round 2 --scene 6
python 4_feature_extract.py --round 2 --scene 6 --slice
python 5_train.py --round 2 --scene 6 --bsz 10 --cp data/round2/s4/result/M0Round2Scene4.pth
python 6_cal_pos.py --round 2 --scene 6
python 7_fix_out_of_range.py --round 2 --scene 6
python 8_fill_back.py --round 2 --scene 6
```
