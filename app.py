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

# Load model
@st.cache_resource
def load_model():
    with open("stacking_classifier_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# Load dummy data
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
        "Academic_Performance_Encoded": np.random.randint(0, 3, size=n),
        "Level": np.random.randint(0, 3, size=n)
    }
    return pd.DataFrame(data)

# ==================== Plotting ====================

def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    return fig

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

# ==================== Sidebar ====================
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman", ["Identitas", "Data Description", "Prediction", "About"])

# ==================== Halaman Identitas ====================
if page == "Identitas":
    st.title("👤 Halaman Identitas Pengguna")
    nama = st.text_input("Masukkan Nama Anda:")
    umur = st.number_input("Masukkan Umur Anda:", min_value=0, max_value=120, value=20)

    if nama.strip():
        st.success(f"Halo {nama}, umur Anda **{umur}** tahun")
        st.session_state["nama"] = nama
        st.session_state["umur"] = umur
    else:
        st.warning("Silakan masukkan nama terlebih dahulu.")

# ==================== Halaman Data ====================
elif page == "Data Description":
    st.title("📊 Deskripsi Data")
    st.write("Dataset simulasi mahasiswa untuk prediksi tingkat stres.")
    data = load_data()
    st.subheader("Cuplikan Data")
    st.dataframe(data)

# ==================== Halaman Prediksi ====================
elif page == "Prediction":
    st.title("📈 Prediksi Tingkat Stres")
    if "nama" in st.session_state and "umur" in st.session_state:
        st.info(f"Prediksi untuk: {st.session_state['nama']}, umur **{st.session_state['umur']} tahun")
    else:
        st.warning("Silakan isi identitas terlebih dahulu di halaman Identitas.")

    st.write("Masukkan informasi berikut untuk prediksi:")

    study_hours = st.slider("Study Hours per Day", 0, 12, 4)
    sleep_hours = st.slider("Sleep Duration (hours/day)", 0, 12, 7)
    physical = st.slider("Physical Activity (hours/week)", 0, 20, 3)
    social = st.slider("Social Hours per Day", 0, 12, 2)
    extracurricular = st.selectbox("Kegiatan Ekstrakurikuler", ["Yes", "No"])
    gpa = st.number_input("GPA", 0.0, 4.0, 3.0)

    extra_binary = 1 if extracurricular == "Yes" else 0
    academic_perf = 2 if gpa >= 3.5 else 1 if gpa >= 2.5 else 0

    input_df = pd.DataFrame([{
        "Study_Hours_Per_Day": study_hours,
        "Sleep_Hours_Per_Day": sleep_hours,
        "Physical_Activity_Hours_Per_Day": physical,
        "Social_Hours_Per_Day": social,
        "Extracurricular_Hours_Per_Day": extra_binary,
        "GPA": gpa,
        "Academic_Performance_Encoded": academic_perf
    }])

    if st.button("Prediksi"):
        try:
            prediction_num = model.predict(input_df)[0]
            prediction_label = map_stress_level(prediction_num)

            st.success(f"Prediksi Tingkat Stres: **{prediction_label}**")

            # ================= Evaluasi Model =================
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
            st.pyplot(plot_confusion_matrix(y, y_pred, labels))

            st.markdown("ROC Curve")
            st.pyplot(plot_roc_curve(y, y_score, classes))

            st.markdown("Precision-Recall Curve")
            st.pyplot(plot_precision_recall_curve(y, y_score, classes))

        except Exception as e:
            st.error(f"❌ Gagal melakukan prediksi: {str(e)}")

# ==================== Halaman About ====================
elif page == "About":
    st.title("ℹ Tentang Aplikasi Ini")
    st.write("""
    Aplikasi ini menggunakan model Stacking Classifier untuk memprediksi tingkat stres mahasiswa berdasarkan:
    - Jam belajar, tidur, aktivitas fisik, sosial, ekstrakurikuler
    - GPA dan performa akademik

    Kategori tingkat stres:
    - 0: Low
    - 1: Moderate
    - 2: High
    """)
