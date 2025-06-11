# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Fungsi: Load model
@st.cache_resource
def load_model():
    with open("stacking_classifier_model.pkl", "rb") as f:
        return pickle.load(f)

# Fungsi: Load data dummy
@st.cache_data
def load_data(n=300):
    np.random.seed(42)
    data = {
        "Study Hours": np.random.randint(0, 10, size=n),
        "Sleep Duration": np.random.randint(4, 10, size=n),
        "Physical Activity": np.random.randint(0, 5, size=n),
        "Social Hours": np.random.randint(0, 6, size=n),
        "Extracurricular Activities": np.random.randint(0, 2, size=n),
        "GPA": np.round(np.random.uniform(2.0, 4.0, size=n), 2),
    }
    return pd.DataFrame(data)

# Inisialisasi
model = load_model()
data = load_data()

# Tentukan fitur yang digunakan oleh model
features = [
    "Study Hours",
    "Sleep Duration",
    "Physical Activity",
    "Social Hours",
    "Extracurricular Activities",
    "GPA"
]

# Prediksi stress level (dengan validasi)
try:
    data["Stress_Level"] = model.predict(data[features])
except Exception as e:
    st.error(f"❌ Gagal memprediksi Stress Level: {e}")
    st.stop()

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Description", "Prediction", "About"])

# Halaman: Data Description
if page == "Data Description":
    st.title("📊 Stress Analysis Dataset Description")

    st.write("""
    Dataset ini berisi informasi tentang tingkat stres mahasiswa yang diukur berdasarkan:
    - **Study Hours**
    - **Sleep Duration**
    - **Physical Activity**
    - **Social Hours**
    - **Extracurricular Activities**
    - **GPA**

    Target variabel adalah **Stress Level** (hasil prediksi model).
    """)

    st.subheader("Data Preview")
    st.dataframe(data.head(20))

    st.subheader("Distribusi Stress Level (Prediksi)")
    fig, ax = plt.subplots()
    sns.countplot(y="Stress_Level", data=data, palette="Set2", ax=ax)
    ax.set_xlabel("Count")
    ax.set_ylabel("Predicted Stress Levels")
    ax.set_title("Distribution of Predicted Stress Levels")
    st.pyplot(fig)

    st.subheader("Feature Distributions (Histogram)")
    for feature in features:
        st.write(f"### {feature}")
        fig, ax = plt.subplots()
        sns.histplot(data[feature], kde=True, bins=20, ax=ax, color='skyblue')
        ax.set_xlabel(feature)
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

    st.subheader("Feature vs Stress Level (Boxplot)")
    for feature in features:
        st.write(f"### {feature} by Stress Level")
        fig, ax = plt.subplots()
        sns.boxplot(x="Stress_Level", y=feature, data=data, palette="pastel", ax=ax)
        st.pyplot(fig)

# Halaman: Prediction
elif page == "Prediction":
    st.title("🔮 Stress Level Prediction")

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
        try:
            prediction = model.predict(input_data[features])[0]
            st.success(f"🎯 Predicted Stress Level: **{prediction}**")
        except Exception as e:
            st.error(f"❌ Gagal memprediksi: {e}")

# Halaman: About
elif page == "About":
    st.title("🧠 About This Model")
    st.write("""
    Model ini menggunakan pendekatan **Stacking Classifier** untuk memprediksi tingkat stres mahasiswa.

    - **Model Base**: Kombinasi dari beberapa algoritma
    - **Model Meta**: Menggabungkan output dari model base
    - **Kelebihan**: Meningkatkan akurasi dan generalisasi prediksi

    Model dilatih dengan data sintetis berdasarkan fitur-fitur seperti durasi belajar, aktivitas fisik, jam tidur, dan GPA.
    """)
