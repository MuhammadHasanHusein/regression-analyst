import streamlit as st
import numpy as np
import joblib

# Load the model
model = joblib.load('logistic_regression_model.sav')

# Title of the web app
st.title('Prediksi Status Berdasarkan Gender, Usia, dan Std')

# Input data from user
st.write("Masukkan data berikut untuk memprediksi status:")

# Create input fields for the features
gender = st.selectbox('Jenis Kelamin (0 = Laki-laki, 1 = Perempuan)', [0, 1])
usia = st.number_input('Usia (Dalam Bulan)', min_value=0, max_value=120, value=18)
std = st.number_input('Berat Badan', min_value=0.0, max_value=120.9, value=2.0)

# Button for prediction
if st.button('Prediksi Status'):
    # Prepare the input data
    input_data = np.array([[gender, usia, std]])
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Debug output to check the prediction value
    st.write(f'Prediction value: {prediction[0]}')
    
    # Map prediction to status
    status_map = {
        0: 'Severely Underweight',
        1: 'Underweight',
        2: 'Normal',
        3: 'Overweight'
    }
    
    # Check if the prediction is within the expected range
    if prediction[0] in status_map:
        status = status_map[prediction[0]]
        # Display result
        st.success(f'Status: {status}')
    else:
        st.error('Nilai prediksi tidak valid. Silakan cek kembali model Anda.')
