import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as SklearnLR

print("=" * 50)
print("WEEK 4 — STUDENT PERFORMANCE PREDICTOR")
print("=" * 50)

X = np.array([1, 2, 3, 4, 5])
y = np.array([20, 40, 50, 65, 80])

print("\n Dataset:")
for i in range(len(X)):
    print(f"   {X[i]} hours studied → {y[i]} marks")

print("\n PART 1: Training from scratch...")

weight = 0.0
bias = 0.0
learning_rate = 0.01
epochs = 1000

for epoch in range(epochs):
    # Step 1 — Make prediction
    y_pred = weight * X + bias

    # Step 2 — Calculate loss (how wrong?)
    loss = np.mean((y_pred - y) ** 2)

    # Step 3 — Calculate direction to improve
    dw = np.mean(2 * (y_pred - y) * X)
    db = np.mean(2 * (y_pred - y))

    # Step 4 — Take small step to improve
    weight -= learning_rate * dw
    bias   -= learning_rate * db

    if epoch % 100 == 0:
        print(f"   Epoch {epoch:4d} | Loss: {loss:.2f} | Weight: {weight:.2f} | Bias: {bias:.2f}")

print(f"\nScratch Training Complete!")
print(f"Weight: {weight:.2f} (1 hour = {weight:.1f} marks)")
print(f"Bias:   {bias:.2f} (0 hours = {bias:.1f} marks baseline)")


print("\n PART 2: Training with Sklearn...")

X_2d = X.reshape(-1, 1) 
sklearn_model = SklearnLR()
sklearn_model.fit(X_2d, y)

print(f"\n Sklearn Training Complete!")
print(f"   Weight: {sklearn_model.coef_[0]:.2f}")
print(f"   Bias:   {sklearn_model.intercept_:.2f}")

print("\n COMPARISON:")
print(f"   {'Hours':<8} {'Real':<8} {'Scratch':<12} {'Sklearn':<12}")
print("   " + "-" * 40)
for h in range(1, 9):
    real = y[h-1] if h <= 5 else "?"
    scratch_pred = weight * h + bias
    sklearn_pred = sklearn_model.predict([[h]])[0]
    print(f"   {h:<8} {str(real):<8} {scratch_pred:<12.1f} {sklearn_pred:<12.1f}")

print("\n JARVIS PREDICTIONS:")
for hours in [6, 7, 8, 10]:
    marks = weight * hours + bias
    print(f"   Study {hours} hours → {marks:.1f} marks predicted")

print("\n Generating visualization...")

y_pred_line = weight * X + bias

plt.figure(figsize=(10, 6))

plt.scatter(X, y, color='blue', s=100, zorder=5, label='Real Data')

plt.plot(X, y_pred_line, color='red', linewidth=2, label='Scratch Model')

X_plot = np.array([1, 2, 3, 4, 5, 6, 7, 8])
sklearn_line = sklearn_model.predict(X_plot.reshape(-1, 1))
plt.plot(X_plot, sklearn_line, color='green', linewidth=2,
         linestyle='--', label='Sklearn Model')

plt.scatter([6, 8], [weight*6+bias, weight*8+bias],
            color='red', s=150, marker='*', zorder=6, label='Predictions')

plt.title(' Jarvis: Student Performance Predictor', fontsize=14)
plt.xlabel('Hours Studied')
plt.ylabel('Marks Scored')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/prediction_chart.png')
plt.show()
print(" Chart saved to results/prediction_chart.png")
print("=" * 50)