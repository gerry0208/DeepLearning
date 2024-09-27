import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def forward(x):
    return w * x + b


def cost(xs, ys):
    sum = 0
    for x, y in zip(xs, ys):
        y_pred = forward(x)
        sum += (y_pred - y) ** 2
    return sum / len(xs)


x_data = [1.0, 2.0, 3.0]
y_data = [5.0, 8.0, 11.0]

W, B = np.arange(0, 4.1, 0.1), np.arange(0, 4.1, 0.1)
w, b = np.meshgrid(W, B)
z = cost(x_data, y_data)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(W, B, z, cmap=cm.Blues)

plt.show()
