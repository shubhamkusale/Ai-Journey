import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv('https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv', 
                 sep='\t', header=None, names=['label', 'message'])

print(df.head())
print(df.shape)
print(df['label'].value_counts())

df['label'] = df['label'].map({'ham':0, 'spam': 1})

X = df['message']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size= 0.2 , random_state= 42)

vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print(X_train_vec.shape)
print(f"training Samples :{X_train.shape[0]}")

model = MultinomialNB()
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)

print("Accuracy :", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


def predict_email(email):
    email_vec = vectorizer.transform([email])
    prediction = model.predict(email_vec)
    return "SPAM" if prediction[0] == 1 else "HAM"


print(predict_email("Congratulations! You won a free iPhone click now"))
print(predict_email("Hey are we meeting tomorrow at 3pm?"))