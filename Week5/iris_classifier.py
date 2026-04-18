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
    
    def predict(self, X):
        predictions = []

        for x in X:
            distances = [self.euclidean_distance(x, x_train)
                        for x_train in self.X_train] 
            k_indices = np.argsort(distances)[:self.k]
            
            k_labels = [self.y_train[i] for i in k_indices]
            prediction = Counter(k_labels).most_common(1)[0][0]

            predictions.append(prediction)
        return np.array(predictions)

knn = KNN(k =3)
knn.fit(X_train, y_train)
prediction = knn.predict(X_test)
print("Predictions:", prediction)
print("Actual:     ", y_test)