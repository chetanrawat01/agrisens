import streamlit as st
import numpy as np
import pandas as pd
import os
import pickle
from PIL import Image

# ✅ Define Paths
BASE_DIR = os.path.dirname(__file__)  # Get the base directory
IMAGE_DIR = os.path.join(BASE_DIR, "CROP-RECOMMENDATION", "crop_images")  # ✅ Updated path
MODEL_PATH = os.path.join(BASE_DIR, "AgriSens\RF.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "AgriSens\label_encoder.pkl")

# ✅ Load Model & Label Encoder
if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)
else:
    st.error("❌ Model or Label Encoder not found! Please train the model first.")
    st.stop()

# ✅ Crop Information
crop_details = {
    "banana": {"description": "🍌 Requires warm and humid regions.", "irrigation": "Weekly", "harvest_time": "10-12 months", "price_trend": "Stable", "best_selling_time": "Year-round", "selling_methods": "Fruit traders, local markets"},
    "rice": {"description": "🌾 Requires warm temperature and high humidity.", "irrigation": "Every 5-7 days", "harvest_time": "120-150 days", "price_trend": "Stable with seasonal spikes", "best_selling_time": "Post-monsoon (Oct-Dec)", "selling_methods": "Local markets, government procurement, online platforms"},
    "maize": {"description": "🌽 Grows well in warm climates with well-drained soil.", "irrigation": "Every 7-10 days", "harvest_time": "90-110 days", "price_trend": "Moderate fluctuations", "best_selling_time": "Pre-summer (Feb-Apr)", "selling_methods": "Wholesale markets, food companies"},
    "chickpea": {"description": "🌱 Prefers cool weather and well-drained loamy soil.", "irrigation": "Every 10-12 days", "harvest_time": "90-120 days", "price_trend": "Stable", "best_selling_time": "Winter (Nov-Feb)", "selling_methods": "Local grain markets, food processors"},
    "kidneybeans": {"description": "🫘 Thrives in moderate temperature with sandy loam soil.", "irrigation": "Every 8-10 days", "harvest_time": "90-120 days", "price_trend": "Moderate fluctuations", "best_selling_time": "Post-rainy season", "selling_methods": "Local markets, grocery suppliers"},
    "pigeonpeas": {"description": "🌿 Requires tropical climate with well-drained soil.", "irrigation": "Every 10-15 days", "harvest_time": "150-180 days", "price_trend": "Stable", "best_selling_time": "Post-harvest (Nov-Jan)", "selling_methods": "Grain markets, bulk buyers"},
    "mothbeans": {"description": "🌱 Grows in dry conditions, needs sandy soil.", "irrigation": "Every 12-15 days", "harvest_time": "70-90 days", "price_trend": "High fluctuations", "best_selling_time": "Winter", "selling_methods": "Local farmers, wholesale traders"},
    "mungbean": {"description": "🌱 Prefers warm climate and sandy loam soil.", "irrigation": "Every 7-10 days", "harvest_time": "60-90 days", "price_trend": "Moderate", "best_selling_time": "Spring", "selling_methods": "Pulses mills, grain traders"},
    "blackgram": {"description": "🌱 Grows well in humid climate.", "irrigation": "Every 10-12 days", "harvest_time": "90-110 days", "price_trend": "Stable", "best_selling_time": "Post-monsoon", "selling_methods": "Wholesale grain markets"},
    "lentil": {"description": "🌿 Thrives in cool temperatures and loamy soil.", "irrigation": "Every 12-15 days", "harvest_time": "100-120 days", "price_trend": "Moderate", "best_selling_time": "Winter", "selling_methods": "Pulses traders"},
    "coffee": {"description": "☕ Prefers cool, humid climates.", "irrigation": "Every 10 days", "harvest_time": "2-3 years", "price_trend": "High volatility", "best_selling_time": "Winter", "selling_methods": "Coffee exporters"},
    "orange": {"description": "🍊 Thrives in tropical climates.", "irrigation": "Every 12-15 days", "harvest_time": "180-210 days", "price_trend": "Moderate fluctuations", "best_selling_time": "Winter", "selling_methods": "Fruit mandis, supermarkets"},
    "papaya": {"description": "🍈 Needs warm temperature and well-drained soil.", "irrigation": "Every 10-12 days", "harvest_time": "150-180 days", "price_trend": "High demand year-round", "best_selling_time": "All seasons", "selling_methods": "Local markets, fruit stores"},
    "mango": {"description": "🥭 Requires warm climate.", "irrigation": "Every 15 days", "harvest_time": "4-5 years", "price_trend": "Seasonal spike", "best_selling_time": "Summer", "selling_methods": "Fruit mandis, online stores"},
    "grapes": {"description": "🍇 Prefers moderate temperatures.", "irrigation": "Every 7-10 days", "harvest_time": "150-180 days", "price_trend": "High fluctuations", "best_selling_time": "Spring", "selling_methods": "Wine industry, export markets"},
    "watermelon": {"description": "🍉 Needs hot, dry climate and sandy soil.", "irrigation": "Every 4-5 days", "harvest_time": "80-100 days", "price_trend": "High in summer", "best_selling_time": "Summer", "selling_methods": "Fruit markets, supermarkets"},
    "muskmelon": {"description": "🍈 Prefers warm weather and sandy loam soil.", "irrigation": "Every 3-4 days", "harvest_time": "75-90 days", "price_trend": "High during peak season", "best_selling_time": "Summer", "selling_methods": "Fruit traders, bulk suppliers"},
    "apple": {"description": "🍏 Grows best in cold regions.", "irrigation": "Every 10-12 days", "harvest_time": "150-180 days", "price_trend": "Seasonal high", "best_selling_time": "Winter", "selling_methods": "Fruit mandis, export traders"},
    "coconut": {"description": "🥥 Grows well in humid coastal climates.", "irrigation": "Every 15-20 days", "harvest_time": "180-240 days", "price_trend": "Stable", "best_selling_time": "Year-round", "selling_methods": "Oil mills, local markets"},
    "jute": {"description": "🧵 Jute is a natural fiber crop grown in hot, humid climates. It requires high rainfall and well-drained sandy loam soil.", "irrigation": "Every 7-10 days", "harvest_time": "120-150 days", "price_trend": "Moderate fluctuations based on global demand", "best_selling_time": "Post-monsoon (Sep-Nov)", "selling_methods": "Textile industries, fiber markets, export companies"},
    "pomegranate": {"description": "🍎 Thrives in hot and dry climates with well-drained soil.","irrigation": "Every 7-10 days","harvest_time": "150-180 days","price_trend": "High demand during festive seasons","best_selling_time": "Autumn and Winter","selling_methods": "Local markets, supermarkets, juice industries"},
    "cotton": {"description": "🌿 Grows well in warm climates with black soil.","irrigation": "Every 15-20 days","harvest_time": "150-180 days","price_trend": "Moderate fluctuations based on global demand","best_selling_time": "Post-harvest (Oct-Jan)","selling_methods": "Textile industries, local markets, export traders"},
}

# ✅ Function to Predict Crop
def predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall):
    feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    input_data = pd.DataFrame([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]], columns=feature_names)

    # ✅ Predict
    predicted_crop = model.predict(input_data)[0]
    predicted_crop_name = label_encoder.inverse_transform([predicted_crop])[0]

    return predicted_crop_name

# ✅ Function to Display Crop Info & Image
def show_crop_info(crop_name):
    col1, col2 = st.columns([1.5, 2.5])
    
    with col1:
        st.markdown(f"<h3 style='color: green;'>🌿 {crop_name.capitalize()} Guide</h3>", unsafe_allow_html=True)
        details = crop_details.get(crop_name.lower(), {})
        if details:
            st.info(details["description"])
            st.markdown(f"**💧 Irrigation:** {details['irrigation']}")
        else:
            st.warning("ℹ No detailed information available for this crop.")

    # ✅ Debug image path
    image_path = os.path.join(IMAGE_DIR, f"{crop_name.lower()}.png")
    st.text(f"🔍 Checking Image Path: {image_path}")  # ✅ Print path for debugging

    with col2:
        if os.path.exists(image_path):
            st.image(Image.open(image_path), caption=f"🌿 Recommended Crop: {crop_name}")
        else:
            st.warning(f"❌ Image not found: {image_path}")

# ✅ Streamlit UI
def main():
    st.markdown("<h2 style='text-align: center; color: #4CAF50;'>🌾 SMART CROP RECOMMENDATION 🌾</h2>", unsafe_allow_html=True)

    # Debugging: Print working directory
    st.text(f"📂 Current Working Directory: {os.getcwd()}")

    # Sidebar Inputs
    st.sidebar.markdown("<h2 style='color: #4CAF50;'>🌱 Enter Soil & Climate Conditions</h2>", unsafe_allow_html=True)

    nitrogen = st.sidebar.number_input("Nitrogen (N)", min_value=0.0, max_value=140.0, value=0.0, step=1.0)
    phosphorus = st.sidebar.number_input("Phosphorus (P)", min_value=0.0, max_value=145.0, value=0.0, step=1.0)
    potassium = st.sidebar.number_input("Potassium (K)", min_value=0.0, max_value=205.0, value=0.0, step=1.0)
    temperature = st.sidebar.number_input("Temperature (°C)", min_value=0.0, max_value=51.0, value=0.0, step=0.1)
    humidity = st.sidebar.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
    ph = st.sidebar.number_input("pH Level", min_value=0.0, max_value=9.0, value=0.0, step=0.1)
    rainfall = st.sidebar.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=0.0, step=1.0)

    # Predict Button
    if st.sidebar.button("🌿 Predict Crop"):
        prediction = predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall)
        st.success(f"🌱 Recommended Crop: **{prediction.capitalize()}**")
        show_crop_info(prediction)

# ✅ Run the App
if __name__ == '__main__':
    main()