import tensorflow as tf
import numpy as np

csi = np.load("data/Round0InputData1_S1.npy", mmap_mode="r")
pos = np.loadtxt("data/Round0InputPos1.txt")[:, 1:]


# 定义一个函数，将NumPy数组转换为tf.train.Example
def create_example(csi_real, csi_imag, pos):
    feature = {
        "csi_real": tf.train.Feature(
            float_list=tf.train.FloatList(value=csi_real.flatten())
        ),
        "csi_imag": tf.train.Feature(
            float_list=tf.train.FloatList(value=csi_imag.flatten())
        ),
        "pos": tf.train.Feature(float_list=tf.train.FloatList(value=pos.flatten())),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


# 将csi的实部和虚部分别存储为csi_real和csi_imag
csi_real = csi.real
csi_imag = csi.imag

# 定义TFRecords文件的路径
tfrecords_file = "data/Round0InputData1_S1.tfrecords"

# 使用tf.io.TFRecordWriter写入数据
with tf.io.TFRecordWriter(tfrecords_file) as writer:
    for i in range(csi.shape[0]):
        example = create_example(csi_real[i], csi_imag[i], pos[i])
        writer.write(example.SerializeToString())

print(f"TFRecords文件已保存到: {tfrecords_file}")
