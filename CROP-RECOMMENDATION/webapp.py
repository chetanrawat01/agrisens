## Importing necessary libraries for the web app
import streamlit as st
import numpy as np
import pandas as pd
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import warnings
from PIL import Image

warnings.filterwarnings('ignore')

# Load dataset
DATA_PATH = "Crop_recommendation.csv"
if not os.path.exists(DATA_PATH):
    st.error(f"Error: {DATA_PATH} file not found! Please upload the dataset.")
    st.stop()

df = pd.read_csv(DATA_PATH)

# Define features and target variable
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model file path
MODEL_PATH = "RF.pkl"

# Load or train model
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
else:
    model = RandomForestClassifier(n_estimators=20, random_state=5)
    model.fit(X_train, y_train)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

# Function to make predictions
def predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall):
    input_data = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
    prediction = model.predict(input_data)
    return prediction[0]

# Function to display crop image
def show_crop_image(crop_name):
    image_path = os.path.join('crop_images', f"{crop_name.lower()}.jpg")
    if os.path.exists(image_path):
        st.image(image_path, caption=f"Recommended crop: {crop_name}", use_column_width=True)
    else:
        st.warning("No image available for this crop.")

# Streamlit app UI
def main():
    st.markdown("<h1 style='text-align: center;'>🌾 SMART CROP RECOMMENDATION 🌾</h1>", unsafe_allow_html=True)
    
    # Sidebar inputs
    st.sidebar.title("AgriSens - Crop Advisor")
    st.sidebar.header("Enter Soil & Climate Conditions")
    
    nitrogen = st.sidebar.number_input("Nitrogen (N)", min_value=0.0, max_value=140.0, value=50.0, step=1.0)
    phosphorus = st.sidebar.number_input("Phosphorus (P)", min_value=0.0, max_value=145.0, value=50.0, step=1.0)
    potassium = st.sidebar.number_input("Potassium (K)", min_value=0.0, max_value=205.0, value=50.0, step=1.0)
    temperature = st.sidebar.number_input("Temperature (°C)", min_value=0.0, max_value=51.0, value=25.0, step=0.1)
    humidity = st.sidebar.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    ph = st.sidebar.number_input("pH Level", min_value=0.0, max_value=14.0, value=6.5, step=0.1)
    rainfall = st.sidebar.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=200.0, step=1.0)

    # Predict button
    if st.sidebar.button("Predict"):
        prediction = predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall)
        st.success(f"🌱 Recommended Crop: **{prediction}**")
        show_crop_image(prediction)

# Run the app
if __name__ == '__main__':
    main()
