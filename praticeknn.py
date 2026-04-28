import numpy as np
class KNN:

    def __init__(self, k= 3):
        self.k = k


    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def eucladean_dis(self, x1 , x2):
        return np.sqrt(np.sum((x1 - x2)**2))
    
    def predict(self, X):
        prediction = []

        for x in X:
            distance = [self.eucladean_dis(x, x_train)
                        for x_train in self.x_train]
            k_indices = np.argsort(distance)[:self.k]
            prediction = Counter(k_labels).most_common(1)[0][0]
            predictions.append(prediction)
        return np.array(predictions)
