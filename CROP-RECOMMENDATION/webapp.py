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

# ✅ Crop Information (Irrigation, Harvest, Price, Selling)
crop_details = {
    "rice": {
        "description": "🌾 Rice needs warm temperatures (20-30°C) and high humidity. Ideal soil: clayey with good water retention.",
        "irrigation": "Every 5-7 days",
        "harvest_time": "120-150 days",
        "price_trend": "Stable with seasonal spikes",
        "best_selling_time": "Post-monsoon (Oct-Dec)",
        "selling_methods": "Local markets, government procurement, online platforms"
    },
    "maize": {
        "description": "🌽 Maize grows well in warm climates (21-27°C) with well-drained sandy loam soil.",
        "irrigation": "Every 7-10 days",
        "harvest_time": "90-110 days",
        "price_trend": "Moderate fluctuations",
        "best_selling_time": "Pre-summer (Feb-Apr)",
        "selling_methods": "Wholesale markets, direct contracts with food companies"
    },
    "wheat": {
        "description": "🌾 Wheat requires a cool climate (15-22°C) and well-drained loamy soil.",
        "irrigation": "Every 10-12 days",
        "harvest_time": "120-150 days",
        "price_trend": "Generally stable with government price support",
        "best_selling_time": "Post-harvest (Mar-May)",
        "selling_methods": "Government mandis, local grain markets, bulk buyers"
    },
    "cotton": {
        "description": "🧺 Cotton requires warm temperatures (25-35°C) and light, well-drained soil.",
        "irrigation": "Every 15 days",
        "harvest_time": "160-180 days",
        "price_trend": "High fluctuations due to international demand",
        "best_selling_time": "Post-harvest (Sep-Nov)",
        "selling_methods": "Textile industries, direct mill contracts"
    },
    "sugarcane": {
        "description": "🍬 Sugarcane needs a hot climate (25-38°C) and well-irrigated soil.",
        "irrigation": "Every 10-12 days",
        "harvest_time": "10-14 months",
        "price_trend": "Stable with government control",
        "best_selling_time": "Year-round (Factory contracts)",
        "selling_methods": "Sugar mills, ethanol production companies"
    },
}

# ✅ Function to Predict Crop
def predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall):
    input_data = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
    return model.predict(input_data)[0]

# ✅ Function to Display Crop Info
def show_crop_info(crop_name):
    image_path = os.path.join(IMAGE_DIR, f"{crop_name.lower()}.png")

    # Layout with two columns
    col1, col2 = st.columns([1.5, 2.5])

    with col1:
        st.markdown(f"<h3 style='color: green;'>🌿 {crop_name.capitalize()} Guide</h3>", unsafe_allow_html=True)
        
        # Fetch crop details
        details = crop_details.get(crop_name.lower(), None)
        if details:
            st.info(details["description"])
            st.markdown(f"**💧 Irrigation:** {details['irrigation']}")
            st.markdown(f"**⏳ Harvest Time:** {details['harvest_time']}")
            st.markdown(f"**📉 Market Price Trends:** {details['price_trend']}")
            st.markdown(f"**🛒 Best Time to Sell:** {details['best_selling_time']}")
            st.markdown(f"**📦 Selling Methods:** {details['selling_methods']}")
        else:
            st.warning("ℹ No detailed information available for this crop.")

    with col2:
        # Show Image (Right Side)
        if os.path.exists(image_path):
            img = Image.open(image_path).resize((400, 300))  # Fixed size
            st.image(img, caption=f"🌿 Recommended Crop: {crop_name}", use_container_width=False)
        else:
            st.warning("⚠ No image available for this crop.")

# ✅ Streamlit App UI
def main():
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🌾 SMART CROP RECOMMENDATION 🌾</h1>", unsafe_allow_html=True)
    
    # Sidebar Inputs
    st.sidebar.markdown("<h2 style='color: #4CAF50;'>🌱 Enter Soil & Climate Conditions</h2>", unsafe_allow_html=True)
    
    nitrogen = st.sidebar.number_input("Nitrogen (N)", min_value=0.0, max_value=140.0, value=50.0, step=1.0)
    phosphorus = st.sidebar.number_input("Phosphorus (P)", min_value=0.0, max_value=145.0, value=50.0, step=1.0)
    potassium = st.sidebar.number_input("Potassium (K)", min_value=0.0, max_value=205.0, value=50.0, step=1.0)
    temperature = st.sidebar.number_input("Temperature (°C)", min_value=0.0, max_value=51.0, value=25.0, step=0.1)
    humidity = st.sidebar.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    ph = st.sidebar.number_input("pH Level", min_value=0.0, max_value=14.0, value=6.5, step=0.1)
    rainfall = st.sidebar.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=200.0, step=1.0)

    # Predict Button
    if st.sidebar.button("🌿 Predict Crop"):
        prediction = predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall)
        st.success(f"🌱 Recommended Crop: **{prediction.capitalize()}**")
        show_crop_info(prediction)

# ✅ Run the App
if __name__ == '__main__':
    main()
