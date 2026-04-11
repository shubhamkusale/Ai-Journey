import pandas as pd
import numpy as np

X = np.array([1, 2, 3, 4, 5])
y = np.array([20, 40, 50, 65, 80])

weight = 0.0
bias = 0.0
learning_rate = 0.01
epoches = 1000

for epoch in range(epoches):
    y_pred = weight * X + bias

    loss = np.mean((y_pred - y)**2)

    dw = np.mean(2 *(y_pred - y) * X)
    db = np.mean(2*(y_pred - y))

    weight -= learning_rate * dw     
    bias -= learning_rate * db


    if epoch % 100 == 0:
        print(f" Weight - {weight :.2f} | bias - {bias: .2f} | epoch - {epoch: .2f} ")

print(f"\nScratch Training Complete!")
print(f"Weight: {weight:.2f} (1 hour = {weight:.1f} marks)")
print(f"Bias:   {bias:.2f} (0 hours = {bias:.1f} marks baseline)")