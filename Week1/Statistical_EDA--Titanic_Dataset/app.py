import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ---------- Scratch Functions ----------

def my_mean(values):
    return sum(values) / len(values)


def my_variance(values):
    mean = my_mean(values)
    total = 0

    for x in values: 
        diff = x - mean
        squared = diff ** 2
        total += squared

    return total / len(values)


def my_std(values):
    return my_variance(values) ** 0.5


# ---------- UI ----------

st.title("📊 Statistical EDA Calculator")
st.write("Upload a CSV file and calculate statistics from scratch.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.write("### Dataset Preview")
    st.dataframe(df.head())

    numeric_columns = df.select_dtypes(include=np.number).columns

    column = st.selectbox("Choose Column", numeric_columns)

    if st.button("Calculate Statistics"):

        values = df[column].dropna().tolist()

        if len(values) == 0:
            st.warning("No valid data in this column")
        else:

            st.write("## Results")

            st.write("### Scratch Implementation")
            st.write(f"Mean: {my_mean(values):.4f}")
            st.write(f"Variance: {my_variance(values):.4f}")
            st.write(f"Std: {my_std(values):.4f}")

            st.write("### NumPy Verification")
            st.write(f"Mean: {np.mean(values):.4f}")
            st.write(f"Variance: {np.var(values):.4f}")
            st.write(f"Std: {np.std(values):.4f}")

            if round(my_std(values), 4) == round(np.std(values), 4):
                st.success("✅ Results Match NumPy!")
            else:
                st.error("❌ Results do not match")

            # ---------- Histogram ----------
            fig, ax = plt.subplots()
            ax.hist(values, bins=20)
            ax.set_title(f"Distribution of {column}")
            ax.set_xlabel(column)
            ax.set_ylabel("Frequency")
            st.pyplot(fig)