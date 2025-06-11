import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

# Load model
@st.cache_resource
def load_model():
    model_path = "/mnt/data/stacking_classifier_model.pkl"
    if not os.path.exists(model_path):
        st.error("❌ File model tidak ditemukan di /mnt/data/. Harap upload ulang.")
        st.stop()
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        st.stop()

# Load dummy data
@st.cache_data
def load_data(model, n=300):
    np.random.seed(42)
    df = pd.DataFrame({
        "Study_Hours_Per_Day": np.random.randint(0, 10, size=n),
        "Sleep_Hours_Per_Day": np.random.randint(4, 10, size=n),
        "Physical_Activity_Hours_Per_Day": np.random.randint(0, 5, size=n),
        "Social_Hours_Per_Day": np.random.randint(0, 6, size=n),
        "Extracurricular_Hours_Per_Day": np.random.randint(0, 2, size=n),
        "GPA": np.round(np.random.uniform(2.0, 4.0, size=n), 2),
        "Academic_Performance_Encoded": np.random.randint(0, 3, size=n)
    })
    try:
        df["Stress_Level"] = model.predict(df)
    except Exception as e:
        st.error(f"❌ Gagal memprediksi Stress Level: {e}")
        st.stop()
    return df

# Muat model dan data
model = load_model()
data = load_data(model)

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Go to", ["Data Description", "Prediction", "About"])

# Data Description
if page == "Data Description":
    st.title("📊 Stress Analysis Dataset Description")
    st.write("Berikut adalah data simulasi mahasiswa beserta prediksi tingkat stres:")

    st.dataframe(data.head())

    st.subheader("Distribusi Stress Level")
    fig, ax = plt.subplots()
    sns.countplot(y="Stress_Level", data=data, palette="Set2", ax=ax)
    st.pyplot(fig)

# Prediction
elif page == "Prediction":
    st.title("🎯 Predict Stress Level")

    input_dict = {
        "Study_Hours_Per_Day": st.slider("Study Hours per Day", 0, 12, 4),
        "Sleep_Hours_Per_Day": st.slider("Sleep Hours per Day", 0, 12, 7),
        "Physical_Activity_Hours_Per_Day": st.slider("Physical Activity per Day", 0, 10, 2),
        "Social_Hours_Per_Day": st.slider("Social Hours per Day", 0, 10, 2),
        "Extracurricular_Hours_Per_Day": st.slider("Extracurricular Hours per Day", 0, 4, 1),
        "GPA": st.number_input("GPA", min_value=0.0, max_value=4.0, value=3.0),
        "Academic_Performance_Encoded": st.selectbox("Academic Performance", [0, 1, 2])  # 0 = Low, 1 = Medium, 2 = High
    }

    input_df = pd.DataFrame([input_dict])

    if st.button("Predict"):
        try:
            prediction = model.predict(input_df)[0]
            st.success(f"Predicted Stress Level: **{prediction}**")
        except Exception as e:
            st.error(f"❌ Gagal memprediksi: {e}")

# About
elif page == "About":
    st.title("🧠 About This App")
    st.write("""
    Aplikasi ini memprediksi tingkat stres mahasiswa berdasarkan:
    - Jam belajar, tidur, aktivitas fisik, sosial, dan ekstrakurikuler
    - GPA dan performa akademik

    Model yang digunakan adalah **Stacking Classifier** dengan beberapa base learner.
    """)
