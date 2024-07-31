import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report


# Muat dan praproses data menggunakan st.cache_data
@st.cache_data
def load_data():
    data = pd.read_excel('heart.xlsx')
    return data


data = load_data()
X = data.drop('target', axis=1)
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = MinMaxScaler(feature_range=(0, 1))
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)

# Bangun model KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_normalized, y_train)
y_pred = knn.predict(X_test_normalized)
accuracy = accuracy_score(y_test, y_pred)

# Definisikan fungsi prediksi
def predict_manual_data(input_data):
    scaled_input = scaler.transform([input_data])
    prediction = knn.predict(scaled_input)
    return 'Memiliki penyakit jantung' if prediction[0] == 1 else 'Tidak memiliki penyakit jantung'

def save_prediction(input_data, result):
    import csv
    import os
    fieldnames = ['Age', 'Sex', 'Chest Pain Type', 'Resting BP', 'Cholesterol', 'Fasting Blood Sugar', 'Resting ECG', 'Max Heart Rate', 'Exercise Angina', 'ST Depression', 'ST Slope', 'Major Vessels', 'Thalassemia', 'Prediction']
    file_exists = os.path.isfile('prediction_history.csv')
    
    with open('prediction_history.csv', 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({**dict(zip(fieldnames[:-1], input_data)), **{'Prediction': result}})

# Antar muka Sidebar yang Ditingkatkan
st.sidebar.title("❤️ WEB Prediksi Penyakit Jantung")
st.sidebar.write("Setiap langkah kecil menuju hidup sehat adalah langkah besar untuk jantungmu.")
st.sidebar.markdown("---")

menu = st.sidebar.radio(
    "📌 Pilih Menu",
    ("Prediksi Manual", "Visualisasi Data", "Riwayat Prediksi")
)

st.sidebar.markdown("---")

if menu == "Prediksi Manual":
    st.sidebar.info("Masukkan data pasien untuk memprediksi penyakit jantung.")
    st.header("👤 Prediksi Penyakit Jantung Secara Manual")

    input_data = [
        st.number_input('Usia', min_value=0, max_value=100, value=25),
        st.selectbox('Jenis Kelamin', [(0, 'Pria'), (1, 'Wanita')], format_func=lambda x: x[1]),
        st.slider('Tipe Nyeri Dada', 0, 3, 1),
        st.number_input('Tekanan Darah Istirahat (mmHg)', min_value=80, max_value=200, value=120),
        st.number_input('Kolesterol Serum (mg/dl)', min_value=100, max_value=400, value=200),
        st.selectbox('Gula Darah Puasa > 120 mg/dl?', [(0, 'Tidak'), (1, 'Ya')], format_func=lambda x: x[1]),
        st.slider('Hasil Elektrokardiografi Istirahat', 0, 2, 1),
        st.number_input('Detak Jantung Maksimal yang Dicapai', min_value=50, max_value=200, value=150),
        st.selectbox('Angin yang Dipicu oleh Olahraga?', [(0, 'Tidak'), (1, 'Ya')], format_func=lambda x: x[1]),
        st.number_input('Depresi ST', min_value=0.0, max_value=10.0, value=1.0),
        st.slider('Kemiringan ST', 0, 2, 1),
        st.slider('Jumlah Pembuluh Darah Utama', 0, 3, 0),
        st.slider('Talasemia', 1, 3, 2)
    ]
    
    if st.button('Prediksi'):
        input_list = [x[0] if isinstance(x, tuple) else x for x in input_data]
        prediction = predict_manual_data(input_list)
        save_prediction(input_list, prediction)  # Simpan ke riwayat
        st.success(f'Prediksi: {prediction}')


elif menu == "Visualisasi Data":
    st.sidebar.info("Jelajahi berbagai visualisasi terkait data penyakit jantung.")
    st.header("📊 Visualisasi Data")
    st.write("Pratinjau Data Set:", data.head())
    st.write(f"Akurasi Model: {accuracy:.2f}%")
    st.text(classification_report(y_test, y_pred))

    # Visualisasi Distribusi Usia
    fig, ax = plt.subplots()
    sns.histplot(data['age'], kde=True, ax=ax)
    ax.set_title('Distribusi Usia Pasien')
    st.pyplot(fig)

    # Bar Chart untuk Jumlah Kasus Berdasarkan Jenis Kelamin
    fig, ax = plt.subplots()
    sns.countplot(x='sex', data=data, ax=ax)
    ax.set_title('Jumlah Kasus Berdasarkan Jenis Kelamin')
    ax.set_xticklabels(['Pria', 'Wanita'])
    st.pyplot(fig)

    # Scatter Plot Tekanan Darah dan Kolesterol
    fig, ax = plt.subplots()
    sns.scatterplot(x='trestbps', y='chol', hue='target', data=data, ax=ax)
    ax.set_title('Hubungan Tekanan Darah dan Kolesterol Terhadap Penyakit Jantung')
    st.pyplot(fig)

elif menu == "Riwayat Prediksi":
    st.sidebar.info("Tinjau log prediksi yang telah lalu.")
    st.header("📜 Riwayat Prediksi")
    try:
        history_data = pd.read_csv('prediction_history.csv')
        st.write(history_data)
    except FileNotFoundError:
        st.error("File riwayat belum ada. Lakukan beberapa prediksi terlebih dahulu.")

