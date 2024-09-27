import numpy as np
import matplotlib.pyplot as plt


def forward(x):
    return w * x


def loss(x, y):
    return (forward(x) - y) ** 2


def gradient(x, y):
    return 2 * x * (forward(x) - y)


x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0
print("Predict (before training)", 4, forward(4))
for epoch in range(100):
    for x, y in zip(x_data, y_data):
        w -= 0.01*gradient(x, y)
        loss_val = loss(x, y)
        print('Epoch: ', epoch, 'w= ', w, 'loss= ', loss_val)
print("Predict (after training)", 4, forward(4))