# 📊 Statistical EDA — Titanic Dataset

### Week 1 of 32 | AI Engineering & Jarvis Roadmap

## 🚀 Project Overview

This project performs **statistical exploratory data analysis (EDA)** on a dataset using **pure Python implementations** of fundamental statistical measures.

The goal of this week was to understand how core mathematical concepts used in machine learning — such as mean, variance, and standard deviation — work internally by implementing them from scratch and validating the results against NumPy.

The project also includes a simple **Streamlit web application** that allows users to upload any CSV dataset and compute statistics interactively.

---

## 🧠 What I Learned

* How mean, variance, and standard deviation are calculated mathematically
* Translating mathematical formulas into Python code
* Working with real datasets using Pandas
* Handling missing values (`dropna`)
* Verifying custom implementations using NumPy
* Building a simple data tool as a web application using Streamlit

---

## 🛠️ Features

* Upload any CSV dataset
* Select numeric column
* Calculate:

  * Mean
  * Variance
  * Standard Deviation
* Compare results with NumPy
* Interactive dataset preview
* Clean web interface

---

## 📂 Project Structure

```
week1-statistical-eda/
│
├── eda.ipynb      # Jupyter notebook with analysis
├── app.py         # Streamlit web app
├── train.csv      # Titanic dataset
└── README.md
```

---

## ▶️ How to Run the Web App

Install dependencies:

```
pip install streamlit pandas numpy matplotlib
```

Run the application:

```
streamlit run app.py
```

Then open the browser link shown in the terminal.

---

## 🤖 Jarvis Roadmap Connection

This project is part of a 32-week journey toward building a personal AI assistant (Jarvis).

Week 1 focuses on statistical foundations that will later be used in machine learning models and AI decision systems.

---

## 📈 Future Improvements

* Data visualization (histograms)
* Additional statistical metrics
* Deployment to cloud platform

---

## 👨‍💻 Author

Shubh — AI Engineering Student
Learning in public and building AI systems step-by-step.
