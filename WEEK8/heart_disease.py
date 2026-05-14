import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

X =df.drop(columns=['target'])
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state= 42, test_size= 0.2)
scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train) 

X_test_scaled = scaler.transform(X_test) 

print("Before scaling - area max :", X_train['mean area'].max())
print("After scaling - area max: ",X_train_scaled[:, 3].max())


# Train Logistic Regression
lr = LogisticRegression(max_iter=10000)
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)

# Train SVM
svm = SVC(kernel='rbf', random_state=42)
svm.fit(X_train_scaled, y_train)
svm_pred = svm.predict(X_test_scaled)

# Compare
print("Logistic Regression:", accuracy_score(y_test, lr_pred))
print("SVM:", accuracy_score(y_test, svm_pred))

print("\n--- Logistic Regression ---")
print(classification_report(y_test, lr_pred))

print("\n--- SVM ---")
print(classification_report(y_test, svm_pred))
print("\n--- Logistic Regression ---")
print(classification_report(y_test, lr_pred))

print("\n--- SVM ---")
print(classification_report(y_test, svm_pred))

cm = confusion_matrix(y_test, lr_pred)
print(cm)

sns.heatmap(cm, annot=True, fmt='d', 
            xticklabels=['Malignant','Benign'],
            yticklabels=['Malignant','Benign'])
plt.title('Confusion Matrix - Logistic Regression')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('confusion_matrix.png')
plt.show()

lr_scores = cross_val_score(LogisticRegression(max_iter=10000),
                                                X_train_scaled, y_train, cv = 5)

print("LR CV srores", lr_scores)
print("LR Average:", lr_scores.mean())
print("LR Std Dev:", lr_scores.std())   