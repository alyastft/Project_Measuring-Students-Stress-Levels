import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, ConfusionMatrixDisplay
from sklearn.preprocessing import label_binarize

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
        "Study_Hours_Per_Day": np.random.randint(0, 10, size=n),
        "Sleep_Hours_Per_Day": np.random.randint(4, 10, size=n),
        "Physical_Activity_Hours_Per_Day": np.random.randint(0, 5, size=n),
        "Social_Hours_Per_Day": np.random.randint(0, 6, size=n),
        "Extracurricular_Hours_Per_Day": np.random.randint(0, 2, size=n),
        "GPA": np.round(np.random.uniform(2.0, 4.0, size=n), 2),
    }
    df = pd.DataFrame(data)
    df["Academic_Performance_Encoded"] = (df["GPA"] >= 3.0).astype(int)
    df["Level"] = np.random.randint(0, 3, size=n)
    return df

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
    n_classes = y_test_bin.shape[1]

    fig, ax = plt.subplots(figsize=(6,5))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        # Mapping prediksi ke label
        label_mapping = {0: "Low", 1: "Moderate", 2: "High"}
        ax.plot(fpr, tpr, lw=2, label=f'{label_mapping.get(classes[i], classes[i])} (AUC = {roc_auc:.2f})')
        
    ax.plot([0,1], [0,1], 'k--', lw=2)
    ax.set_xlim([0,1])
    ax.set_ylim([0,1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc='lower right')
    return fig

# Fungsi untuk plotting Precision-Recall Curve
def plot_precision_recall_curve(y_true, y_score, classes):
    y_test_bin = label_binarize(y_true, classes=classes)
    n_classes = y_test_bin.shape[1]

    fig, ax = plt.subplots(figsize=(6,5))
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
        pr_auc = auc(recall, precision)
        # Mapping prediksi ke label
        label_mapping = {0: "Low", 1: "Moderate", 2: "High"}
        ax.plot(recall, precision, lw=2, label=f'{label_mapping.get(classes[i], classes[i])} (AUC = {pr_auc:.2f})')

    ax.set_xlim([0,1])
    ax.set_ylim([0,1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc='lower left')
    return fig

# Sidebar Navigation
st.sidebar.title("Navigasi")
page = st.sidebar.selectbox("Select Page", ["Identity", "Data Description", "Prediction", "About Models"])

# ===================== Halaman Identitas =====================
if page == "Identity":
    st.title("👤 User Identity Page")

    # Input nama dan umur
    nama = st.text_input("Input Your Name:")
    umur = st.number_input("Input Your Age:", min_value=0, max_value=120, value=20)

    if nama.strip():
        st.success(f"Hai {nama}, you're {umur} years old")
        # Simpan di session state agar bisa digunakan di halaman lain
        st.session_state["nama"] = nama
        st.session_state["umur"] = umur
    else:
        st.warning("Please input the name first.")

# ===================== Halaman Data Description =====================
elif page == "Data Description":
    st.title("📊 Students' Stress Level Dataset Description")
    st.write("""
    This dataset contains information about student stress levels as measured by:
    - Study Hours
    - Sleep Duration
    - Physical Activity
    - Social Hours
    - Extracurricular Activities
    - GPA
    
    The target variable is Stress Level.
    """)

    data = load_data()
    
    # Mapping kolom Level menjadi label kategori
    label_mapping = {0: "Low", 1: "Moderate", 2: "High"}
    data["Level"] = data["Level"].map(label_mapping)

    st.subheader("Here's the raw data")
    st.dataframe(data)

    # Tambahkan visualisasi distribusi stress level
    st.subheader("Class distribution")
    st.bar_chart(data["Level"].value_counts().reindex(["Low", "Moderate", "High"]))

    # Pie Chart Distribusi Tingkat Stres
    st.subheader("Distribution of Stress Level (Pie Chart)")
    stress_counts = data["Level"].value_counts().reindex(["Low", "Moderate", "High"])

    fig, ax = plt.subplots()
    colors = ["#66b3ff", "#ffcc99", "#ff9999"]  # Warna untuk Low, Moderate, High
    ax.pie(
        stress_counts,
        labels=stress_counts.index,
        autopct="%1.1f%%",
        startangle=140,
        colors=colors,
        explode=(0.05, 0.05, 0.05),
        shadow=True,
    )
    ax.set_title("Distribution of Students' Stress Level")
    ax.axis("equal")  # Agar lingkarannya proporsional
    st.pyplot(fig)

    data = load_data()
    
    # Pastikan mapping dilakukan SEBELUM scatter plot
    label_mapping = {0: "Low", 1: "Moderate", 2: "High"}
    data["Level"] = data["Level"].map(label_mapping)
    
    # Cek isi data
    st.write("🔍 Debugging Data Sample:", data.head())
    
    # Scatter plot
    st.subheader("Scatter Plot: GPA vs Study Hours")
    fig_scatter, ax = plt.subplots()
    
    colors = {"Low": "#66b3ff", "Moderate": "#ffcc99", "High": "#ff9999"}
    
    for level in ["Low", "Moderate", "High"]:
        subset = data[data["Level"] == level]
        ax.scatter(
            subset["Study_Hours_Per_Day"],
            subset["GPA"],
            label=level,
            alpha=0.7,
            edgecolors='k',
            s=80,
            c=colors[level]
        )
    
    ax.set_xlabel("Study Hours Per Day")
    ax.set_ylabel("GPA")
    ax.set_title("Relationship between Study Hours and GPA based on Stress Levels")
    ax.legend(title="Stress Level")
    
    st.pyplot(fig_scatter)



# ===================== Halaman Prediction =====================
elif page == "Prediction":
    st.title("📈 Predicted Stress Level")

    if "nama" in st.session_state and "umur" in st.session_state:
        st.info(f"Prediction for: {st.session_state['nama']}, {st.session_state['umur']} years old")
    else:
        st.warning("Please fill in your identity first on the Identity page.")

    st.write("Enter the following information to predict stress levels:")

    study_hours = st.slider("Study Hours per Day", 0, 12, 4)
    sleep_duration = st.slider("Sleep Duration per Day (hours)", 0, 12, 7)
    physical_activity = st.slider("Physical Activity (hours/week)", 0, 20, 3)
    social_hours = st.slider("Social Hours per Day", 0, 12, 2)
    extracurricular = st.selectbox("Participate in extracurricular activities?", ["Yes", "No"])
    gpa = st.number_input("GPA", min_value=0.0, max_value=4.0, value=3.0)

    # Konversi ekstrakurikuler ke binary
    extracurricular_binary = 1 if extracurricular == "Yes" else 0

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
            
            # Mapping prediksi ke label
            label_mapping = {0: "Low", 1: "Moderate", 2: "High"}
            predicted_label = label_mapping.get(prediction, "Unknown")

            if "nama" in st.session_state:
                st.success(f"{st.session_state['nama']}, your stress levels are predicted: **{predicted_label}**")
            else:
                st.success(f"Stress level predicted: **{predicted_label}**")

            # Evaluasi model
            st.markdown("---")
            st.subheader("Evaluation of Models with Dummy Data")

            data = load_data()
            X = data.drop(columns=["Level"])
            y = data["Level"]
            classes = [0, 1, 2]
            class_labels = ["Low", "Moderate", "High"]


            expected_columns = [
                "Study_Hours_Per_Day",
                "Extracurricular_Hours_Per_Day",
                "Sleep_Hours_Per_Day",
                "Social_Hours_Per_Day",
                "Physical_Activity_Hours_Per_Day",
                "GPA",
                "Academic_Performance_Encoded"
            ]
            X = X[expected_columns]
            
            y_pred = model.predict(X)
            y_score = model.predict_proba(X)

            st.markdown("Confusion Matrix")
            fig_cm = plot_confusion_matrix(y, y_pred, class_labels)
            st.pyplot(fig_cm)

            st.markdown("ROC Curve")
            fig_roc = plot_roc_curve(y, y_score, classes)
            st.pyplot(fig_roc)

            st.markdown("Precision-Recall Curve")
            fig_pr = plot_precision_recall_curve(y, y_score, classes)
            st.pyplot(fig_pr)

        except Exception as e:
            st.error(f"An error occurs during prediction: {str(e)}")

# ===================== Halaman About =====================
elif page == "About Models":
    st.title("ℹ About This Model")
    st.write("""
    This model uses the Stacking Classifier approach to predict students' stress levels. 
    Stacking is an ensemble machine learning method that combines multiple base and meta models to improve accuracy.

    - Base Model: A combination of several algorithms
    - Meta Model: Combining the outputs of the base models
    - Pros: Improves accuracy and generalization
    """)
