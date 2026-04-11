import numpy as np




class KNN:
    def __init__(self, k = 3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def euclidean_distance(self, x1, x2):
        return (np.sqrt(np.sum(x1 - x2)**2))