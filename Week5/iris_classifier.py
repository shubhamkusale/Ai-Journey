import numpy as np
from collections import Counter
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

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
    
    def Manhattan_distance(self, x1, x2):
        return np.sum(np.abs(x1 - x2))
    
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
# Accuracy Score
accuracy = accuracy_score(y_test, prediction)
print("Accuracy:", accuracy)
# Confusion Matrix
cm = (confusion_matrix(y_test, prediction))
print("confusion_matrix:")
print(cm)
# Elbow Method
k_values = range(1, 20, 2)  
accuracies = []

for k in k_values:
    knn = KNN(k=k)
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
    acc = accuracy_score(y_test, pred)
    accuracies.append(acc)
    print(f"K={k} → Accuracy={acc*100:.1f}%")

    # Visualization - only using 2 features
X_vis = X[:, :2]

# Split again with 2 features only
X_train_vis, X_test_vis, y_train_vis, y_test_vis = train_test_split(
    X_vis, y, test_size=0.2, random_state=42
)

# Train KNN on 2 features
knn_vis = KNN(k=3)
knn_vis.fit(X_train_vis, y_train_vis)

# Create mesh grid
x_min, x_max = X_vis[:, 0].min() - 1, X_vis[:, 0].max() + 1
y_min, y_max = X_vis[:, 1].min() - 1, X_vis[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.arange(x_min, x_max, 0.1),
    np.arange(y_min, y_max, 0.1)
)

# Predict every point in mesh
Z = knn_vis.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X_vis[:, 0], X_vis[:, 1], c=y, edgecolors='black')
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("KNN Decision Boundary - Iris Dataset")
plt.show()