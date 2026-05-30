import streamlit as st
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
from sklearn.model_selection import train_test_split

@st.cache_resource
def load_model():
    df = pd.read_csv('https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv',sep='\t',header=None, names = ['label','message'])
    df['label'] = df['label'].map({'ham' : 0, 'spam' : 1})
    X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)
    vectorizer = CountVectorizer()
    x_train_vec = vectorizer.fit_transform(X_train)
    model = MultinomialNB()
    model.fit(x_train_vec, y_train)
    return model, vectorizer

model, vectorizer = load_model()

st.title("🚨 Spam Detector")
st.subheader("Paste any SMS or email below")
user_input = st.text_area("Your message:", height=150)

if st.button("Check Message"):
    if user_input.strip() == "":
        st.warning("Please enter a message first.")
    else:
        vec = vectorizer.transform([user_input])
        prediction = model.predict(vec)[0]
        probability = model.predict_proba(vec)[0]
        
        if prediction == 1:
            st.error(f"🚨 SPAM detected! Confidence: {probability[1]*100:.1f}%")
        else:
            st.success(f"✅ Looks safe! Ham confidence: {probability[0]*100:.1f}%")

st.markdown("---")
st.caption("Built by Shubham Kusale · github.com/shubhamkusale")