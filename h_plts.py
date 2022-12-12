import numpy as np
import matplotlib.pyplot as plt
from utils.add_bodies import BOX_SIZE, GAP

lower_bound = -BOX_SIZE[1] - GAP / 2
upper_bound = BOX_SIZE[1] + GAP / 2
N = 100

# y_range is the list of bin edges
y_range = np.linspace(lower_bound, upper_bound, N, endpoint=False)
h = np.zeros_like(y_range)
h_bar = 2 * 0.1 / np.pi * np.arctan(np.sin(2. * np.pi * 10. * y_range + 1./4 * np.pi/0.1) / 0.001) + 0.15

for i in range(N):
    if -GAP / 2 < y_range[i] < GAP / 2:
        h[i] = 0.25
    else:
        h[i] = 0.05

'''
plt.title("h(x)")
plt.xlabel("x (end effector position)")
plt.ylabel("h(x) (distance from environment to end effector)")
plt.axvline(x=0, c="black")
plt.axhline(y=0, c="black")
plt.plot(y_range, h)
plt.show()
'''
plt.title(r"$\bar{h}(x)$")
plt.xlabel("x (end effector position)")
plt.ylabel("h(x) (distance from environment to end effector)")
plt.axvline(x=0, c="black")
plt.axhline(y=0, c="black")
plt.plot(y_range, h_bar)
plt.show()