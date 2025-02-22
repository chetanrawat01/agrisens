import streamlit as st
import numpy as np
import pandas as pd
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from PIL import Image

# ✅ Fix: Get absolute paths
DATA_PATH = os.path.join(os.path.dirname(__file__), "Crop_recommendation.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "RF.pkl")
IMAGE_DIR = os.path.join(os.path.dirname(__file__), "crop_images")

# ✅ Ensure the dataset exists
if not os.path.exists(DATA_PATH):
    st.error(f"❌ Error: {DATA_PATH} file not found! Please upload the dataset.")
    st.stop()

# ✅ Load dataset
df = pd.read_csv(DATA_PATH)

# Define features and labels
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ✅ Load or Train Model
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
else:
    model = RandomForestClassifier(n_estimators=20, random_state=5)
    model.fit(X_train, y_train)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

# ✅ Crop Guidance Data
crop_details = {
    "rice": "🌾 Rice needs warm temperatures (20-30°C) and high humidity. Ideal soil: clayey with good water retention.",
    "maize": "🌽 Maize grows well in warm climates (21-27°C) with well-drained sandy loam soil.",
    "chickpea": "🌱 Chickpeas prefer cool weather (10-25°C) and well-drained loamy soil with neutral pH.",
    "kidneybeans": "🫘 Kidney beans thrive in temperatures around 15-25°C and well-drained sandy loam soil.",
    "pigeonpeas": "🌿 Pigeon peas grow best in tropical climates (18-30°C) with light, well-drained soil.",
    "mothbeans": "🌱 Moth beans need dry conditions (25-35°C) and sandy soil with low water availability.",
    "mungbean": "🌱 Mung beans require warm temperatures (25-35°C) and well-drained sandy loam soil.",
    "blackgram": "🌱 Black gram prefers warm, humid conditions (25-30°C) with well-drained loamy soil.",
    "lentil": "🌿 Lentils thrive in cool temperatures (10-25°C) and light loamy soil.",
    "pomegranate": "🍎 Pomegranates prefer hot, dry climates (25-35°C) and loamy soil with good drainage.",
    "banana": "🍌 Bananas grow best in warm, humid regions (25-30°C) with rich, well-drained soil.",
    "mango": "🥭 Mango trees require warm climates (25-35°C) and well-drained sandy loam soil.",
    "grapes": "🍇 Grapes grow well in moderate temperatures (15-30°C) with loamy, well-drained soil.",
    "watermelon": "🍉 Watermelons need hot, dry climates (25-35°C) with sandy loam soil and good drainage.",
    "muskmelon": "🍈 Muskmelons prefer warm weather (25-35°C) and sandy loam soil.",
    "apple": "🍏 Apples grow best in cold regions (5-20°C) with well-drained loamy soil.",
    "orange": "🍊 Oranges thrive in tropical climates (15-30°C) with well-drained sandy soil.",
    "papaya": "🍈 Papayas need warm temperatures (25-35°C) and well-drained sandy loam soil.",
    "coconut": "🥥 Coconuts grow well in humid coastal climates (27-32°C) with sandy loam soil.",
    "cotton": "🧺 Cotton requires warm temperatures (25-35°C) and light, well-drained soil.",
    "jute": "🧵 Jute needs hot, humid conditions (24-37°C) with sandy loam soil and high water availability.",
    "coffee": "☕ Coffee plants grow best in cool, humid climates (15-25°C) with well-drained soil."
}

# ✅ Function to Predict Crop
def predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall):
    input_data = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
    return model.predict(input_data)[0]

# ✅ Function to Display Crop Image & Details in Two Columns
def show_crop_info(crop_name):
    image_path = os.path.join(IMAGE_DIR, f"{crop_name.lower()}.png")

    # Layout with two columns
    col1, col2 = st.columns([1, 2])

    with col1:
        # Show Guidance Info (Left Side)
        if crop_name in crop_details:
            st.success(f"🌿 **{crop_name.capitalize()} Guide**")
            st.info(crop_details[crop_name])

    with col2:
        # Show Image (Right Side)
        if os.path.exists(image_path):
            img = Image.open(image_path).resize((400, 300))  # Fixed size
            st.image(img, caption=f"🌿 Recommended Crop: {crop_name}", use_container_width=False)
        else:
            st.warning("⚠ No image available for this crop.")

    # Additional Detailed Section Below
    st.subheader(f"📌 More About {crop_name.capitalize()}")
    st.write("For further details on growing conditions, pests, and harvesting techniques, visit our **[Crop Guide](guide/index.html)**.")

# ✅ Streamlit App UI
def main():
    st.markdown("<h1 style='text-align: center;'>🌾 SMART CROP RECOMMENDATION 🌾</h1>", unsafe_allow_html=True)
    
    # Sidebar Inputs
    st.sidebar.title("AgriSens - Crop Advisor")
    st.sidebar.header("Enter Soil & Climate Conditions")
    
    nitrogen = st.sidebar.number_input("Nitrogen (N)", min_value=0.0, max_value=140.0, value=50.0, step=1.0)
    phosphorus = st.sidebar.number_input("Phosphorus (P)", min_value=0.0, max_value=145.0, value=50.0, step=1.0)
    potassium = st.sidebar.number_input("Potassium (K)", min_value=0.0, max_value=205.0, value=50.0, step=1.0)
    temperature = st.sidebar.number_input("Temperature (°C)", min_value=0.0, max_value=51.0, value=25.0, step=0.1)
    humidity = st.sidebar.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    ph = st.sidebar.number_input("pH Level", min_value=0.0, max_value=14.0, value=6.5, step=0.1)
    rainfall = st.sidebar.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=200.0, step=1.0)

    # Predict Button
    if st.sidebar.button("Predict"):
        prediction = predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall)
        st.success(f"🌱 Recommended Crop: **{prediction}**")
        show_crop_info(prediction)

# ✅ Run the App
if __name__ == '__main__':
    main()
