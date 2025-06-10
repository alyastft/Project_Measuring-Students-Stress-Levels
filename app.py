# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Load model
with open("stacking_classifier_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load data
@st.cache_data
def load_data():
    return pd.read_pickle("students_stress_data.pkl")

@st.cache_resource
def load_model():
    with open("stacking_classifier_model.pkl", "rb") as f:
        return pickle.load(f)

data = load_data()
model = load_model()

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Description", "Prediction", "About"])

# Page 1: Data Description
if page == "Data Description":
    st.title("Data Description")
    st.write("""
    Dataset ini berisi informasi tentang tingkat stres mahasiswa yang diukur berdasarkan:
    - **Study Hours**
    - **Sleep Duration**
    - **Physical Activity**
    - **Social Hours**
    - **Extracurricular Activities**
    - **GPA**
    
    Target variabel adalah **Stress Level**.
    """)

    st.subheader("Data Preview")
    st.dataframe(data.head())

    st.subheader("Stress Level Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x="Stress Level", data=data, ax=ax)
    st.pyplot(fig)

# Page 2: Prediction
elif page == "Prediction":
    st.title("Stress Level Prediction")

    st.write("Masukkan informasi berikut untuk memprediksi tingkat stres mahasiswa:")

    study_hours = st.slider("Study Hours per Day", 0, 12, 4)
    sleep_duration = st.slider("Sleep Duration per Day (hours)", 0, 12, 7)
    physical_activity = st.slider("Physical Activity (hours/week)", 0, 20, 3)
    social_hours = st.slider("Social Hours per Day", 0, 12, 2)
    extracurricular = st.selectbox("Participate in Extracurricular Activities?", ["Yes", "No"])
    gpa = st.number_input("GPA", min_value=0.0, max_value=4.0, value=3.0)

    extracurricular_binary = 1 if extracurricular == "Yes" else 0

    input_data = pd.DataFrame([{
        "Study Hours": study_hours,
        "Sleep Duration": sleep_duration,
        "Physical Activity": physical_activity,
        "Social Hours": social_hours,
        "Extracurricular Activities": extracurricular_binary,
        "GPA": gpa
    }])

    if st.button("Predict"):
        prediction = model.predict(input_data)[0]
        st.success(f"Predicted Stress Level: **{prediction}**")

# Page 3: About
elif page == "About":
    st.title("About This Model")

    st.write("""
    Model ini menggunakan pendekatan **Stacking Classifier** untuk memprediksi tingkat stres mahasiswa. 
    Stacking adalah metode ensemble machine learning yang menggabungkan beberapa model dasar (seperti Random Forest, Logistic Regression, dan SVM) dan memanfaatkan model meta untuk meningkatkan performa prediksi.

    - **Model Base**: Kombinasi dari beberapa algoritma
    - **Model Meta**: Menggabungkan output dari model base
    - **Kelebihan**: Meningkatkan akurasi dan generalisasi prediksi
    """)

    st.markdown("Model ini dilatih menggunakan data historis mahasiswa dan faktor-faktor penentu stres seperti durasi belajar, tidur, aktivitas fisik, dan GPA.")

