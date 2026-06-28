import numpy as np 

X = np.array([[1], [2], [6], [8]])
y = np.array([[0], [0], [1], [1]])

np.random.seed(42)
w = np.random.randn(1,1)
b = np.zeros((1,1))
lr = 0.1


def sigmoid(n):
    return 1 / (1 + np.exp(-n))

for epoch in range(1000):
    z =np.dot(X, w) + b
    a = sigmoid(z)

    print("Guesses", a)

    loss = np.mean((a - y)**2)
    print("loss", loss)

    error = a - y
    d_W = np.dot(X.T,error) / len(X)
    d_b = np.mean(error)

    w = w - lr * d_W
    b = b - lr * d_b

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

print("\nFinal guesses:")
print(sigmoid(np.dot(X, w) + b))