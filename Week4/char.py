import matplotlib.pyplot as plt
import numpy as np

X = np.array([1, 2, 3, 4, 5])
y = np.array([20, 40, 50, 65, 80])

# Your trained weight and bias
weight = 14.53
bias = 7.39

# Predicted line
y_pred = weight * X + bias

# Plot
plt.figure(figsize=(8, 5))
plt.scatter(X, y, color='blue', label='Real Data')
plt.plot(X, y_pred, color='red', label='Predicted Line')
plt.title('Student Performance Predictor')
plt.xlabel('Hours Studied')
plt.ylabel('Marks Scored')
plt.legend()
plt.savefig('results/prediction_chart.png')
plt.show()
print("Chart saved!")