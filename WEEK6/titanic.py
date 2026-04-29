import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
# Load dataset
df = sns.load_dataset('titanic')

# Explore
print(df.shape)
print(df.head())
print(df.info())
print(df.isnull().sum())

df = df[['survived','pclass','sex','age','sibsp','parch','fare','embarked']]

df['age'] = df['age'].fillna(df['age'].median())
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])

print(df.isnull().sum())
print(df.shape)

df['sex'] = df['sex'].map({'male': 0, 'female': 1})
df['embarked'] = df['embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Verify everything is numbers now
print(df.head())
print(df.dtypes)

X = df.drop(columns=['survived'])
y = df['survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 42)

print(X_train.shape)
print(X_test.shape)

print("=== DATA READY FOR TRAINING ===")
print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")
print(f"Features: {list(X.columns)}")


#Training all threee model knn, decision tree random forest

dt =DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train, y_train)

rf = RandomForestClassifier(n_estimators=100, random_state= 42)
rf.fit(X_train, y_train)

knn= KNeighborsClassifier(n_neighbors= 5)
knn.fit(X_train, y_train)

dt_pred = dt.predict(X_test)
rf_pred = rf.predict(X_test)
knn_pred = knn.predict(X_test)\
    
print("Decision Tree:", accuracy_score(y_test, dt_pred))
print("Random Forest:", accuracy_score(y_test, rf_pred))
print("KNN          :", accuracy_score(y_test, knn_pred))