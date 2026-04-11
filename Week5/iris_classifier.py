import numpy as np
from collections import Counter
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()

X = iris.data   
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state= 42, test_size= 0.2
)

print("Training Flowers", X_train.shape)
print("Testing flower", X_test.shape)

class KNN:
    def __init__(self, k = 3):
        self.k = k
        
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def euclidean_distance(self, x1, x2):
        return (np.sqrt(np.sum((x1 - x2)**2)))
    
    def predict(self, x):
        predictions = []

        for x in X:
            distances = [self.euclidean_distance(X , X_train)
                         for X_train in self.X_train]
            
            k_indices = np.argsort(distances)[:self.k]
