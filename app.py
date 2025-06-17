import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.metrics import (
    accuracy_score, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, classification_report
)
from sklearn.preprocessing import label_binarize

# ===========================
# 1. Load Model & Scaler
# ===========================
@st.cache_resource
def load_model_and_scaler():
    with open("stacking_classifier_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model_and_scaler()

# ===========================
# 2. Load Dataset
# ===========================
@st.cache_data
def load_data():
    df = pd.read_csv("student_lifestyle_dataset.csv")
    stress_mapping = {'Low': 0, 'Moderate': 1, 'High': 2}
    performance_mapping = {'Poor': 0, 'Fair': 1, 'Good': 2, 'Excellent': 3}

    df['Academic_Performance'] = df['GPA'].apply(
        lambda x: 'Excellent' if x >= 3.5 else 'Good' if x >= 3.0 else 'Fair' if x >= 2.0 else 'Poor'
    )
    df['Academic_Performance_Encoded'] = df['Academic_Performance'].map(performance_mapping)
    df['Stress_Level_Encoded'] = df['Stress_Level'].map(stress_mapping)
    
    return df

data = load_data()

features = [
    "Study_Hours_Per_Day", "Sleep_Hours_Per_Day", "Physical_Activity_Hours_Per_Day",
    "Social_Hours_Per_Day", "Extracurricular_Hours_Per_Day", "GPA", "Academic_Performance_Encoded"
]

# ===========================
# 3. Sidebar Navigation
# ===========================
st.sidebar.title("Navigasi")
page = st.sidebar.selectbox("Pilih halaman", ["Deskripsi Data", "Evaluasi Model", "Prediksi"])

# ===========================
# 4. Deskripsi Data
# ===========================
if page == "Deskripsi Data":
    st.title("📊 Deskripsi Dataset")
    st.markdown("""
    ## 🗂️ Dataset Overview: Student Lifestyle and Stress
    
    Dataset ini berisi informasi gaya hidup mahasiswa dan hubungannya dengan tingkat stres serta performa akademik.
    
    ### 🎯 Tujuan
    Memprediksi **Stress Level** berdasarkan atribut gaya hidup dan akademik.
    
    ### 🔑 Fitur Utama
    - **Jumlah data:** 2.000 mahasiswa
    - **Kolom:** 8 fitur + target
    - **Fitur gaya hidup:**
      - 🕒 `Study_Hours_Per_Day`: Jam belajar per hari
      - 😴 `Sleep_Hours_Per_Day`: Jam tidur per hari
      - 🏃‍♂️ `Physical_Activity_Hours_Per_Day`: Aktivitas fisik harian
      - 🗣️ `Social_Hours_Per_Day`: Interaksi sosial
      - 🎭 `Extracurricular_Hours_Per_Day`: Kegiatan ekstrakurikuler
    - **Akademik & Target:**
      - 🎓 `GPA`: Nilai rata-rata akademik
      - ⚡ `Stress_Level`: Target prediksi — Low, Moderate, High
    
    ### 📌 Insight Data
    - Stres tinggi → Jam belajar tinggi & tidur rendah
    - Stres rendah → Aktivitas fisik & sosial seimbang
    - Fitur paling berpengaruh: **Study Hours** & **Sleep Hours**
    """)

    st.dataframe(data.head())

    st.subheader("Distribusi Kelas Stress Level")
    st.markdown("""
    ### 📊 Penjelasan Distribusi Kelas Stress Level
    
    Distribusi ini menunjukkan **jumlah mahasiswa** dalam setiap kategori stres:
    
    - 🟢 **Low**: Mahasiswa yang memiliki gaya hidup seimbang—cukup tidur, waktu belajar moderat, dan aktif secara fisik & sosial.
    - 🟡 **Moderate**: Umumnya memiliki tekanan akademik atau waktu belajar tinggi, namun masih menjaga keseimbangan aktivitas lainnya.
    - 🔴 **High**: Cenderung disebabkan oleh jam belajar berlebihan, kurang tidur, dan minim aktivitas sosial atau fisik.
    
    Distribusi kelas ini penting karena:
    - Memberi gambaran apakah data seimbang atau tidak.
    - Mempengaruhi performa model prediksi (model bisa bias jika mayoritas data berasal dari satu kelas).
    
    💡 Jika proporsi kelas tidak seimbang (misalnya sebagian besar Moderate), teknik seperti **SMOTE** digunakan saat training untuk menyeimbangkan data.
    """)
    st.bar_chart(data["Stress_Level"].value_counts())
    st.markdown("""
    Distribusi ini menunjukkan jumlah mahasiswa yang tergolong dalam tiga tingkat stress:
    - **High** (Tinggi): 1029 mahasiswa (**51.5%**)
    - **Moderate** (Sedang): 674 mahasiswa (**33.7%**)
    - **Low** (Rendah): 297 mahasiswa (**14.9%**)
    
    Distribusi ini menunjukkan bahwa **lebih dari setengah mahasiswa mengalami stres tinggi**, yang mungkin berkaitan dengan tekanan akademik, kurang tidur, atau kebiasaan gaya hidup yang tidak seimbang.
    """)

    st.subheader("Diagram Pie Stress Level")
    fig, ax = plt.subplots()
    data["Stress_Level"].value_counts().plot.pie(
        autopct="%1.1f%%", startangle=90, ax=ax, shadow=True, explode=[0.05]*3
    )
    ax.set_ylabel("")
    st.pyplot(fig)

# ===========================
# 5. Evaluasi Model
# ===========================
elif page == "Evaluasi Model":
    st.title("📈 Evaluasi Model")
    st.markdown("""
    ### 🧪 Evaluasi Model yang Digunakan
    
    Model yang digunakan adalah **Stacking Classifier**, yaitu gabungan dari beberapa model dasar (XGBoost, Logistic Regression, Decision Tree, Random Forest, dan SVM) yang dipadukan menggunakan meta-model Random Forest.
    Untuk menilai performa model, berikut metrik evaluasi yang digunakan:
    
    - **🎯 Akurasi**: Proporsi data uji yang berhasil diprediksi dengan benar. Metrik ini memberikan gambaran umum seberapa sering model membuat prediksi yang benar.
    
    - **📊 Confusion Matrix**: Menunjukkan perbandingan antara label sebenarnya dan hasil prediksi. Memudahkan untuk melihat kesalahan spesifik antar kelas stres (Low, Moderate, High).
    
    - **📉 ROC Curve dan AUC (Area Under Curve)**:
      - ROC (Receiver Operating Characteristic) menunjukkan trade-off antara True Positive Rate dan False Positive Rate.
      - AUC mengukur kemampuan model membedakan antara kelas: semakin tinggi (mendekati 1), semakin baik performa model.
    
    - **🧾 Classification Report**:
      - **Precision**: Seberapa akurat model saat memprediksi suatu kelas.
      - **Recall**: Seberapa baik model mendeteksi semua instance dari suatu kelas.
      - **F1-Score**: Harmonic mean dari precision dan recall, menggambarkan keseimbangan antara keduanya.
    
    Evaluasi dilakukan menggunakan **data asli** yang telah diseimbangkan menggunakan **SMOTE** dan dinormalisasi dengan **RobustScaler**, agar hasil prediksi lebih adil dan tidak bias terhadap kelas dominan.
    """)


    X = data[features]
    X = pd.DataFrame(X, columns=scaler.feature_names_in_)
    X_scaled = scaler.transform(X)

    y = data["Stress_Level_Encoded"]
    class_labels = ["Low", "Moderate", "High"]
    
    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)
    acc = accuracy_score(y, y_pred)

    st.subheader("🎯 Akurasi")
    st.success(f"Akurasi: {acc * 100:.2f}%")

    st.subheader("📊 Confusion Matrix")
    fig_cm, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y, y_pred, display_labels=class_labels, ax=ax)
    st.pyplot(fig_cm)

    st.subheader("📉 ROC Curve")
    y_bin = label_binarize(y, classes=[0, 1, 2])
    fig_roc, ax = plt.subplots()
    for i in range(3):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{class_labels[i]} (AUC = {roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    st.pyplot(fig_roc)

    st.subheader("🧾 Classification Report")
    st.dataframe(pd.DataFrame(classification_report(y, y_pred, target_names=class_labels, output_dict=True)).T)

# ===========================
# 6. Prediksi
# ===========================
elif page == "Prediksi":
    st.title("🔮 Prediksi Tingkat Stres Mahasiswa")

    # 👤 Input identitas di sini (bukan halaman terpisah)
    nama = st.text_input("Nama Anda:")
    umur = st.number_input("Umur Anda:", min_value=5, max_value=100, value=20)

    # 🧾 Input fitur prediksi
    study = st.slider("Jam Belajar per Hari", 0, 12, 4)
    sleep = st.slider("Jam Tidur per Hari", 0, 12, 7)
    activity = st.slider("Aktivitas Fisik per Hari", 0, 5, 2)
    social = st.slider("Jam Sosialisasi per Hari", 0, 6, 2)
    extracurricular = st.selectbox("Ikut Ekstrakurikuler?", ["Ya", "Tidak"])
    gpa = st.number_input("GPA", 0.0, 4.0, 3.2)

    academic_perf = 'Excellent' if gpa >= 3.5 else 'Good' if gpa >= 3.0 else 'Fair' if gpa >= 2.0 else 'Poor'
    performance_encoded = {'Poor': 0, 'Fair': 1, 'Good': 2, 'Excellent': 3}[academic_perf]

    input_df = pd.DataFrame([{
        "Study_Hours_Per_Day": study,
        "Sleep_Hours_Per_Day": sleep,
        "Physical_Activity_Hours_Per_Day": activity,
        "Social_Hours_Per_Day": social,
        "Extracurricular_Hours_Per_Day": 1 if extracurricular == "Ya" else 0,
        "GPA": gpa,
        "Academic_Performance_Encoded": performance_encoded
    }])[features]

    input_df = pd.DataFrame(input_df, columns=scaler.feature_names_in_)
    input_scaled = scaler.transform(input_df)

    if st.button("Prediksi"):
        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0]
        label = {0: "Low", 1: "Moderate", 2: "High"}[pred]

        st.success(f"Hai {nama} (umur {umur}), tingkat stresmu diprediksi: **{label}**")

        st.subheader("📊 Probabilitas Prediksi")
        fig, ax = plt.subplots()
        ax.bar(["Low", "Moderate", "High"], prob, color=["green", "orange", "red"])
        ax.set_ylabel("Probabilitas")
        ax.set_ylim(0, 1)
        st.pyplot(fig)
