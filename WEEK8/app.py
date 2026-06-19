import streamlit as st
import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer

# load the trained model + scaler we saved
model = joblib.load('cancer_model.pkl')
scaler = joblib.load('cancer_scaler.pkl')

# get feature names + a real sample to use as "example"
data = load_breast_cancer()
feature_names = data.feature_names

st.title("Breast Cancer Classifier")
st.write("Predicts whether a tumor is Malignant or Benign from 30 cell measurements.")
st.write("Built by Shubham Kusale — SVM model, 98% accuracy.")

# example button: fills a real benign sample
if st.button("Load Example Tumor"):
    st.session_state.example = data.data[20]   # row 20 is a real sample

# build 30 number inputs
inputs = []
for i, name in enumerate(feature_names):
    default = float(st.session_state.example[i]) if 'example' in st.session_state else 0.0
    val = st.number_input(name, value=default, format="%.4f")
    inputs.append(val)

# predict button
if st.button("Predict"):
    x = np.array(inputs).reshape(1, -1)      # shape it for the model
    x_scaled = scaler.transform(x)           # scale like training
    pred = model.predict(x_scaled)[0]        # 0 or 1
    result = "Benign (not cancer)" if pred == 1 else "Malignant (cancer)"
    st.subheader(f"Prediction: {result}")