import numpy as np


class LinearRegression:
    def __init__(self, lr = 0.01, epoches = 1000):
        self.weight = 0.0
        self.bias = 0.0
        self.learning_rate = lr
        self.epoch = epoches

    def fit(self, X, y):
        for epoces in range(self.epoch):
            y_pred = self.weight * X + self.bias
            loss = np.mean((y_pred - y)** 2)
            dw = np.mean(2*(y_pred - y)* X)
            db = np.mean(2*(y_pred - y))
            self.weight -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            if epoces %100 == 0:
                print(f"epoches :{epoces}, loss {loss:.2f}")
            
    def predict(self, X):
        return self.weight * X + self.bias

X = np.array([1, 2, 3, 4, 5])
y = np.array([20, 40, 50, 65, 80])

model = LinearRegression()
model.fit(X, y)

print(f"6 hours → {model.predict(6):.1f} marks")
print(f"8 hours → {model.predict(8):.1f} marks")