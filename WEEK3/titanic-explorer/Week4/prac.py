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

    dw = np.mean(2 *(y_pred - y) *X)
    db = np.mean(2 *(y_pred - y))

    weight = weight - learning_rate * dw
    bias = bias - learning_rate * db

    if epoch % 100 == 0:
        print(f"epoches {epoch:4d}| Weight {weight:.2f}| Bias {bias:.2f}")

print("Traning Complete ")
print(f"Weight: {weight:.2f}")
print(f"Bias: {bias:.2f}")    
print(f"Study 6 hours → {weight * 6 + bias:.1f} marks")
print(f"Study 8 hours → {weight * 8 + bias:.1f} marks")
