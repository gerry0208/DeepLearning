import numpy as np
import matplotlib.pyplot as plt


def forward(x):
    return w * x


def cost(xs, ys):
    sum = 0
    for x, y in zip(xs, ys):
        y_pred = forward(x)
        sum += (y - y_pred) ** 2
    return sum / len(xs)


def gradient(xs, ys):
    grad = 0
    for x, y in zip(xs, ys):
        y_pred = forward(x)
        grad += 2 * (y_pred - y) * x
    return grad / len(xs)


x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0
print("Predict (before training)", 4, forward(4))
for epoch in range(100):
    w -= 0.01 * gradient(x_data, y_data)
    cost_val = cost(x_data, y_data)
    print("Epoch:", epoch, "w=", w, "cost=", cost_val)
print("Predict (after training)", 4, forward(4))

