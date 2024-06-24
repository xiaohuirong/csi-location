import numpy as np
import matplotlib.pyplot as plt

H = np.load("data/sample0.npy")

# h (sample_num, port_num, ant_num, sc_num)

h = H[0, ...]

h_ang = np.angle(h).reshape(-1)
h_amp = np.abs(h).reshape(-1)

plt.hist(h_amp, bins=50, alpha=0.75, edgecolor="black")
plt.hist(h_ang, bins=50, alpha=0.5, edgecolor="black")

plt.title("Distribution of H")
plt.xlabel("Value")
plt.ylabel("Frequency")

plt.show()
