#!/bin/python

import numpy as np
import matplotlib.pyplot as plt

pos = np.loadtxt("data/Round0InputPos1.txt")

index = pos[:, 0]

X = pos[:, 1]
Y = pos[:, 2]

plt.scatter(X, Y)

plt.show()
