import numpy as np
import matplotlib.pyplot as plt

# H (bsz, port_num, ant_num, sc_num)
H = np.load("data/Round0InputData1.npy", mmap_mode="r")

input_pos = np.loadtxt("data/Round0InputPos1.txt")

index = input_pos[:, 0].astype(int)
x = input_pos[:, 1]
y = input_pos[:, 2]
print(index)

# h (sample_num, port_num, ant_num, sc_num)
print(H.shape)

h = H[index - 1]

# h = np.fft.fftshift(h, axes=-1)
# h = np.fft.ifft(h, axis=-1)
# h = np.fft.fftshift(h, axes=-1)
np.save("data/Round0InputData1_S1_new.npy", h)

h_ang = np.angle(h).reshape(-1)
h_amp = np.abs(h).reshape(-1)

plt.hist(h_amp, bins=50, alpha=0.75, edgecolor="black")
plt.hist(h_ang, bins=50, alpha=0.5, edgecolor="black")

plt.title("Distribution of H")
plt.xlabel("Value")
plt.ylabel("Frequency")

plt.show()

exit()
