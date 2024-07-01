import numpy as np
from show_distance import show_distance
import matplotlib.pyplot as plt

h = np.load("data/Round0InputData1_S1.npy", mmap_mode="r")[0:10]
pos = np.loadtxt("data/Round0InputPos1.txt")[0:10, 1:]

h = np.fft.ifft(h)
h = np.mean(np.abs(h), axis=(1, 2))

print(np.argmax(h[0]))
print(np.argmax(np.roll(h[4], -6)))

plt.plot(h[0])
plt.plot(np.roll(h[4], -6))

plt.show()

h = h.reshape(10, 1, 408)
h_T = np.roll(h.reshape(1, 10, 408), -6)
h_dist = np.square(np.sum(h * h_T, axis=2))

h = h.reshape(10, 408)
h_2 = np.sum(np.square(h), axis=1)
h_2 = h_2.reshape(10, 1)
h_2_T = h_2.reshape(1, 10)
norm = h_2 * h_2_T
h_dist = h_dist / norm

# calculate physical distance
pos = pos.reshape(10, 1, 2)
pos_T = pos.reshape(1, 10, 2)
dist = np.linalg.norm(pos - pos_T, axis=2)

print(h_dist)
print(dist)

pos = pos.reshape(10, 2)
show_distance(h_dist, pos, 0)
