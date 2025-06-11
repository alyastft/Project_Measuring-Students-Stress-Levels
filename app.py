import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, ConfusionMatrixDisplay
from sklearn.preprocessing import label_binarize

# ==================== Fungsi Utama ====================

# Mapping label angka ke kategori stres
def map_stress_level(label):
    return {0: "Low", 1: "Moderate", 2: "High"}.get(label, "Unknown")

# Fungsi untuk load model
@st.cache_resource
def load_model():
    with open("stacking_classifier_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# Fungsi untuk load data dummy
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
        "Level": np.random.randint(0, 3, size=n),
    }
    return pd.DataFrame(data)

# Fungsi untuk plotting Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(ax=ax, cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    return fig

# Fungsi untuk plotting ROC Curve
def plot_roc_curve(y_true, y_score, classes):
    y_test_bin = label_binarize(y_true, classes=classes)
    fig, ax = plt.subplots(figsize=(6,5))
    for i in range(len(classes)):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f'{map_stress_level(classes[i])} (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc='lower right')
    return fig

def plot_precision_recall_curve(y_true, y_score, classes):
    y_test_bin = label_binarize(y_true, classes=classes)
    fig, ax = plt.subplots(figsize=(6,5))
    for i in range(len(classes)):
        precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
        pr_auc = auc(recall, precision)
        ax.plot(recall, precision, lw=2, label=f'{map_stress_level(classes[i])} (AUC = {pr_auc:.2f})')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc='lower left')
    return fig

# Sidebar Navigation
st.sidebar.title("Navigasi")
page = st.sidebar.selectbox("Pilih Halaman", ["Identitas", "Data Description", "Prediction", "About"])

# ===================== Halaman Identitas =====================
if page == "Identitas":
    st.title("👤 Halaman Identitas Pengguna")

    # Input nama dan umur
    nama = st.text_input("Masukkan Nama Anda:")
    umur = st.number_input("Masukkan Umur Anda:", min_value=0, max_value=120, value=20)

    if nama.strip():
        st.success(f"Halo {nama}, umur Anda {umur} tahun")
        # Simpan di session state agar bisa digunakan di halaman lain
        st.session_state["nama"] = nama
        st.session_state["umur"] = umur
    else:
        st.warning("Silakan masukkan nama terlebih dahulu.")

# ===================== Halaman Data Description =====================
elif page == "Data Description":
    st.title("📊 Students' Stress Level Dataset Description")
    st.write("""
    Dataset ini berisi informasi tentang tingkat stres mahasiswa yang diukur berdasarkan:
    - Study Hours
    - Sleep Duration
    - Physical Activity
    - Social Hours
    - Extracurricular Activities
    - GPA
    
    Target variabel adalah Stress Level.
    """)

    data = load_data()
    st.subheader("Cuplikan Data")
    st.dataframe(data)

# ===================== Halaman Prediction =====================
elif page == "Prediction":
    st.title("📈 Prediction of Stress Levels")

    if "nama" in st.session_state and "umur" in st.session_state:
        st.info(f"Prediksi untuk: {st.session_state['nama']}, umur **{st.session_state['umur']} tahun")
    else:
        st.warning("Silakan isi identitas terlebih dahulu di halaman Identitas.")

    st.write("Masukkan informasi berikut untuk memprediksi tingkat stres:")

    study_hours = st.slider("Study Hours per Day", 0, 12, 4)
    sleep_duration = st.slider("Sleep Duration per Day (hours)", 0, 12, 7)
    physical_activity = st.slider("Physical Activity (hours/week)", 0, 20, 3)
    social_hours = st.slider("Social Hours per Day", 0, 12, 2)
    extracurricular = st.selectbox("Ikut Kegiatan Ekstrakurikuler?", ["Yes", "No"])
    gpa = st.number_input("GPA", min_value=0.0, max_value=4.0, value=3.0)

    # Konversi ekstrakurikuler ke binary
    extracurricular_binary = 1 if extracurricular == "Yes" else 0
    academic_perf = 2 if gpa >= 3.5 else 1 if gpa >= 2.5 else 0

    input_data = pd.DataFrame([{
    "Study_Hours_Per_Day": study_hours,
    "Sleep_Hours_Per_Day": sleep_duration,
    "Physical_Activity_Hours_Per_Day": physical_activity,
    "Social_Hours_Per_Day": social_hours,
    "Extracurricular_Hours_Per_Day": extracurricular_binary,
    "GPA": gpa,
    "Academic_Performance_Encoded": 1 if gpa >= 3.0 else 0  # atau sesuai logika yang dipakai saat training
}])

    expected_columns = [
    "Study_Hours_Per_Day",
    "Extracurricular_Hours_Per_Day",
    "Sleep_Hours_Per_Day",
    "Social_Hours_Per_Day",
    "Physical_Activity_Hours_Per_Day",
    "GPA",
    "Academic_Performance_Encoded"
    ]
    input_data = input_data[expected_columns]

    input_data = input_data.astype({
    "Study_Hours_Per_Day": int,
    "Sleep_Hours_Per_Day": int,
    "Physical_Activity_Hours_Per_Day": int,
    "Social_Hours_Per_Day": int,
    "Extracurricular_Hours_Per_Day": int,
    "GPA": float,
    "Academic_Performance_Encoded": int
})

    # Tombol Prediksi
    if st.button("Prediksi"):
        try:
            prediction = model.predict(input_data)[0]
            prediction_label = map_stress_level(prediction)
            if "nama" in st.session_state:
                st.success(f"{st.session_state['nama']}, tingkat stres kamu diprediksi: {prediction}")
            else:
                st.success(f"Tingkat stres diprediksi: {prediction_label}")

            # Evaluasi model
            st.markdown("---")
            st.subheader("Evaluasi Model dengan Data Dummy")

            data = load_data()
            X = data.drop(columns=["Level"])
            y = data["Level"]

            y_pred = model.predict(X)
            y_score = model.predict_proba(X)

            classes = [0, 1, 2]
            labels = [map_stress_level(i) for i in classes]

            st.markdown("Confusion Matrix")
            fig_cm = plot_confusion_matrix(y, y_pred, labels)
            st.pyplot(fig_cm)

            st.markdown("ROC Curve")
            fig_roc = plot_roc_curve(y, y_score, classes)
            st.pyplot(fig_roc)

            st.markdown("Precision-Recall Curve")
            fig_pr = plot_precision_recall_curve(y, y_score, classes)
            st.pyplot(fig_pr)

        except Exception as e:
            st.error(f"Terjadi kesalahan saat prediksi: {str(e)}")

# ===================== Halaman About =====================
elif page == "About":
    st.title("ℹ Tentang Model Ini")
    st.write("""
    Model ini menggunakan pendekatan Stacking Classifier untuk memprediksi tingkat stres mahasiswa. 
    Stacking adalah metode ensemble machine learning yang menggabungkan beberapa model dasar dan meta untuk meningkatkan akurasi.

    - Model Base: Kombinasi dari beberapa algoritma
    - Model Meta: Menggabungkan output dari model base
    - Kelebihan: Meningkatkan akurasi dan generalisasi
    """)
