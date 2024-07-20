import numpy as np

s_3 = np.sqrt(3)
A = [[200, 0], [0, 200]]
P1 = [[-100 * s_3, -100], [100 * s_3, -100], [0, 200]]
P2 = [[100 * s_3, -100], [0, 200], [-100 * s_3, -100]]


def turn_to_square(r, s, pos):
    if r == 0:
        P = P1
    elif r == 1:
        P = P2

    i = s - 1
    p1 = P[i % 3]
    p2 = P[(i + 1) % 3]
    B = np.zeros((2, 2))
    B[:, 0] = p1
    B[:, 1] = p2
    print(B)
    B = np.linalg.inv(B)
    T = np.dot(A, B)
    t_pos = np.dot(T, pos.T).T

    return t_pos


def turn_back(r, s, pos):
    if r == 0:
        P = P1
    elif r == 1:
        P = P2

    i = s - 1
    p1 = P[i % 3]
    p2 = P[(i + 1) % 3]
    B = np.zeros((2, 2))
    B[:, 0] = p1
    B[:, 1] = p2
    print(B)
    i_A = np.linalg.inv(A)
    T = np.dot(B, i_A)
    t_pos = np.dot(T, pos.T).T

    return t_pos
