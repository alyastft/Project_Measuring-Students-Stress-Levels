import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Load model
@st.cache_resource
def load_model():
    with open("/mnt/data/stacking_classifier_model.pkl", "rb") as f:
        return pickle.load(f)

# Load dummy data sesuai format pelatihan model
@st.cache_data
def load_data(n=300):
    np.random.seed(42)
    data = {
        "Study_Hours_Per_Day": np.random.randint(0, 10, size=n),
        "Sleep_Hours_Per_Day": np.random.randint(4, 10, size=n),
        "Physical_Activity_Hours_Per_Day": np.random.randint(0, 5, size=n),
        "Social_Hours_Per_Day": np.random.randint(0, 6, size=n),
        "Extracurricular_Hours_Per_Day": np.random.randint(0, 2, size=n),
        "GPA": np.round(np.random.uniform(2.0, 4.0, size=n), 2),
        "Academic_Performance_Encoded": np.random.randint(0, 3, size=n)
    }
    df = pd.DataFrame(data)
    df["Stress_Level"] = model.predict(df)
    return df

# Load model dan data
model = load_model()
data = load_data()

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Go to", ["Data Description", "Prediction", "About"])

# Halaman 1: Data Description
if page == "Data Description":
    st.title("📊 Stress Analysis Dataset Description")
    st.write("""
    Dataset ini berisi informasi tentang tingkat stres mahasiswa berdasarkan:
    - Study Hours per Day
    - Sleep Hours per Day
    - Physical Activity Hours per Day
    - Social Hours per Day
    - Extracurricular Hours per Day
    - GPA
    - Academic Performance Encoded
    """)
    
    st.subheader("Data Preview")
    st.dataframe(data.head(20))

    st.subheader("Distribusi Stress Level (Prediksi)")
    fig, ax = plt.subplots()
    sns.countplot(y="Stress_Level", data=data, palette="Set2", ax=ax)
    st.pyplot(fig)

    st.subheader("Feature Distributions")
    features = data.columns[:-1]
    for feature in features:
        st.write(f"### {feature}")
        fig, ax = plt.subplots()
        sns.histplot(data[feature], kde=True, bins=20, ax=ax, color='skyblue')
        st.pyplot(fig)

    st.subheader("Feature vs Stress Level (Boxplot)")
    for feature in features:
        st.write(f"### {feature} by Stress Level")
        fig, ax = plt.subplots()
        sns.boxplot(x="Stress_Level", y=feature, data=data, palette="pastel", ax=ax)
        st.pyplot(fig)

# Halaman 2: Prediction
elif page == "Prediction":
    st.title("🎯 Predict Stress Level")

    st.write("Masukkan informasi berikut untuk memprediksi tingkat stres mahasiswa:")

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
        prediction = model.predict(input_df)[0]
        st.success(f"Predicted Stress Level: **{prediction}**")

# Halaman 3: About
elif page == "About":
    st.title("🧠 About This Model")
    st.write("""
    Model ini menggunakan **Stacking Classifier** untuk memprediksi tingkat stres mahasiswa.
    
    - Menggabungkan beberapa model untuk hasil yang lebih akurat
    - Dilatih dengan fitur-fitur seperti jam belajar, aktivitas fisik, tidur, dan GPA
    """)
