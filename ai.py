# Import necessary libraries
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report


# Title and header
st.title('Klasifikasi Penyakit Jantung')
st.header('Prediksi Penyakit Jantung Menggunakan K-Nearest Neighbors')



@st.cache_data
def load_data():
    data = pd.read_excel('heart.xlsx')
    return data


data = load_data()
st.write("Preview of Dataset:", data.head())

# Data preprocessing
X = data.drop('target', axis=1)
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature scaling
scaler = MinMaxScaler(feature_range=(0, 1))
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)

# Model building
knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(X_train_normalized, y_train)

# Predictions
y_pred = knn.predict(X_test_normalized)

# Accuracy and classification report
accuracy = accuracy_score(y_test, y_pred) * 100
st.write(f'Akurasi model: {accuracy:.2f}%')
st.text("Laporan Klasifikasi:")
st.text(classification_report(y_test, y_pred))

# Manual prediction setup
def predict_manual_data(input_data):
    input_array = np.array(input_data).reshape(1, -1)
    prediction = knn.predict(input_array)
    return 'Memiliki penyakit jantung' if prediction[0] == 1 else 'Tidak memiliki penyakit jantung'

# Descriptions for input fields
feature_descriptions = {
    "age": "Usia pasien",
    "sex": "Jenis kelamin (0 = Pria, 1 = Wanita)",
    "cp": "Jenis nyeri dada (0-3)",
    "trestbps": "Tekanan darah saat istirahat (mmHg)",
    "chol": "Kadar Kolestrol serum dalam mg/dl",
    "fbs": "Gula darah puasa > 120 mg/dl? (1 = Ya, 0 = Tidak)",
    "restecg": "Hasil elektrokardiografi (0, 1, 2)",
    "thalach": "Detak jantung maksimum yang dicapai",
    "exang": "Mengalami angin setelah berolahraga? (1 = Ya, 0 = Tidak)",
    "oldpeak": "Depresi Segmen ST (dibandingkan saat istirahat)",
    "slope": "Gradien/Kemiringan Segmen ST",
    "ca": "Jumlah pembuluh darah utama (0-3) diwarnai flourosopi",
    "thal": "Thalasemia: 1 = normal; 2 = cacat; 3 = carrier"
}

# User input features with descriptions
st.header("Prediksi Manual")
input_features = {}

# Manually configure each input
input_features['age'] = st.number_input('age (Usia pasien)', min_value=0, max_value=200, format="%d")
input_features['sex'] = st.number_input('sex (Jenis kelamin (0 = Pria, 1 = Wanita))', min_value=0, max_value=1, format="%d")
input_features['cp'] = st.number_input('cp (Jenis nyeri dada (0-3))', min_value=0, max_value=3, format="%d")
input_features['trestbps'] = st.number_input('trestbps (Tekanan darah saat istirahat (mmHg))', min_value=0, max_value=300, format="%d")
input_features['chol'] = st.number_input('chol (Kadar Kolestrol serum dalam mg/dl)', min_value=0, max_value=600, format="%d")
input_features['fbs'] = st.number_input('fbs (Gula darah puasa > 120 mg/dl? (1 = Ya, 0 = Tidak))', min_value=0, max_value=1, format="%d")
input_features['restecg'] = st.number_input('restecg (Hasil elektrokardiografi (0, 1, 2))', min_value=0, max_value=2, format="%d")
input_features['thalach'] = st.number_input('thalach (Detak jantung maksimum yang dicapai)', min_value=0, max_value=250, format="%d")
input_features['exang'] = st.number_input('exang (Mengalami angin setelah berolahraga? (1 = Ya, 0 = Tidak))', min_value=0, max_value=1, format="%d")
input_features['oldpeak'] = st.number_input('oldpeak (Depresi Segmen ST (dibandingkan saat istirahat))', min_value=0.0, max_value=10.0, format="%.1f")
input_features['slope'] = st.number_input('slope (Gradien/Kemiringan Segmen ST)', min_value=0, max_value=2, format="%d")
input_features['ca'] = st.number_input('ca (Jumlah pembuluh darah utama (0-3) diwarnai flourosopi)', min_value=0, max_value=3, format="%d")
input_features['thal'] = st.number_input('thal (Thalasemia: 1 = normal; 2 = cacat; 3 = carrier)', min_value=1, max_value=3, format="%d")

# Prediction button and result display
if st.button('Predict'):
    # Assuming predict_manual_data function exists and is correctly implemented
    prediction = predict_manual_data(list(input_features.values()))
    st.success(f'Prediksi: {prediction}')


