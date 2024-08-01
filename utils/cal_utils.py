import numpy as np
from scipy.stats import mode


s_3 = np.sqrt(3)
A = [[200, 0], [0, 200]]
P1 = [[-100 * s_3, -100], [100 * s_3, -100], [0, 200]]
P2 = [[100 * s_3, -100], [0, 200], [-100 * s_3, -100]]


def turn_to_square(r, s, pos):
    if r == 0:
        P = P1
    else:
        P = P2

    i = s - 1
    p1 = P[i % 3]
    p2 = P[(i + 1) % 3]
    B = np.zeros((2, 2))
    B[:, 0] = p1
    B[:, 1] = p2
    B = np.linalg.inv(B)
    T = np.dot(A, B)
    t_pos = np.dot(T, pos.T).T

    return t_pos


def turn_back(r, s, pos):
    if r == 0:
        P = P1
    else:
        P = P2

    i = s - 1
    p1 = P[i % 3]
    p2 = P[(i + 1) % 3]
    B = np.zeros((2, 2))
    B[:, 0] = p1
    B[:, 1] = p2
    i_A = np.linalg.inv(A)
    T = np.dot(B, i_A)
    t_pos = np.dot(T, pos.T).T

    return t_pos


def rotate_points(points, angle_degrees):
    angle_radians = np.radians(angle_degrees)
    rotation_matrix = np.array(
        [
            [np.cos(angle_radians), np.sin(angle_radians)],
            [-np.sin(angle_radians), np.cos(angle_radians)],
        ]
    )
    rotated_points = np.dot(points, rotation_matrix.T)
    return rotated_points


def fill_out_with_mean(data):
    # 计算四分位数
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1

    # 计算上下限
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # 去除离群点
    filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]

    mean_data = np.mean(filtered_data)

    data[(data >= lower_bound) & (data <= upper_bound)] = mean_data

    return data


def cal_aoa_tof_feature(h, r, s):
    """
    h : (bsz, 2, 64, 408)
    """
    bsz = h.shape[0]

    abs_h = np.abs(h)

    all_max_index_2 = np.argmax(abs_h[..., :67], -1).reshape(bsz, 2 * 64)

    max_index2 = mode(all_max_index_2, axis=-1).mode

    max_index = max_index2

    tobsz = np.arange(bsz)

    h_max = h[tobsz, :, :, max_index]

    h_max = h_max.reshape(bsz, 4, 8, 4)

    angle = np.angle(h_max)

    angle_diff = -((np.diff(angle, axis=2) + np.pi) % (2 * np.pi) - np.pi) / np.pi
    angle_diff = angle_diff.reshape(bsz, 4 * 7 * 4)

    angle_diff2 = -((np.diff(angle, axis=3) + np.pi) % (2 * np.pi) - np.pi) / (
        4 * np.pi
    )
    angle_diff2 = angle_diff2.reshape(bsz, 4 * 8 * 3)

    aoa = []
    aop = []

    mean_aop = []
    mean_aoa = []

    for i in range(bsz):
        f_data = fill_out_with_mean(angle_diff[i])
        f_data2 = fill_out_with_mean(angle_diff2[i])
        aoa.append(f_data)
        aop.append(f_data2)

        mean_aoa.append(np.mean(f_data))
        mean_aop.append(np.mean(f_data2))

    mean_aoa = np.arccos(np.array(mean_aoa))
    mean_aop = np.arccos(np.array(mean_aop))
    mean_tof = max_index2 * 299.792458 / (408 * 0.24)
    mean_x = mean_tof * np.cos(mean_aoa)
    mean_y = mean_tof * np.sin(mean_aoa)

    mean_values = np.column_stack(
        (
            mean_aoa / np.pi,
            mean_aop / np.pi,
            max_index2 / 67.0,
            mean_x / 200.0,
            mean_y / 200.0,
        )
    )

    aoa = np.arccos(np.array(aoa)) / np.pi
    aop = np.arccos(np.array(aop)) / np.pi

    all_feature = np.concatenate(
        (aoa, aop, all_max_index_2 / 67.0, mean_values), axis=-1
    )

    return all_feature


def rotate_center_to_y(pos, s, r):
    if r == 2 or r == 1:
        if s == 6 or s == 3:
            pos = rotate_points(pos, 180)
        elif s == 5 or s == 2:
            pos = rotate_points(pos, 60)
        elif s == 4 or s == 1:
            pos = rotate_points(pos, -60)

    return pos


def rotate_center_back(pos, s, r):
    if r == 2:
        if s == 6 or s == 3:
            pos = rotate_points(pos, -180)
        elif s == 5 or s == 2:
            pos = rotate_points(pos, -60)
        elif s == 4 or s == 1:
            pos = rotate_points(pos, 60)

    return pos
