import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)

# index = [1144, 1897, 10, 20]
# index = [1144, 1897, 530]
# index = [1144, 1897]
# index = [1110, 1571, 978]
# index = [1110, 978]
index = [0, 4]
# index = [13]
num = len(index)
# h (2, port_num, ant_num, sc_num)
h = np.load("data/Round0InputData1_S1_new.npy", mmap_mode="r")[index]
pos = np.loadtxt("data/Round0InputPos1.txt")[index]

print(pos)
h = np.fft.ifft(h, axis=-1)

sum_h = np.sum(abs(h), axis=(1, 2))
max_index = np.argmax(sum_h, axis=-1)
print(max_index)

# h = h[[0, 1], 0, :, [49, 32]]
h = h.reshape(num, 2, 2, 8, 4, 408)

h1 = h[np.arange(num), 0, 0, :, 0, max_index]
# h2 = h[np.arange(num), 1, 0, :, :, max_index]

fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
# fig, ax = plt.subplots()
# for i in range(num):
#     ax.plot(np.abs(h[i, 0, 0, 0, 0]))
# plt.show()
# exit()

h1_ang = np.angle(h1)
h1_abs = np.abs(h1)
for i in range(num):
    ax.scatter(h1_ang[i], h1_abs[i])

print(h1_ang[0] - h1_ang[1])

# h2_ang = np.angle(h2)
# h2_abs = np.abs(h2)
# for i in range(0, 1):
#     ax.scatter(h2_ang[i], h2_abs[i])

plt.show()
