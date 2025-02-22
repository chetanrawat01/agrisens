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

# ✅ Complete Crop Details
crop_details = {
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
}

# ✅ Function to Predict Crop
def predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall):
    input_data = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
    return model.predict(input_data)[0]

# ✅ Function to Display Crop Info
def show_crop_info(crop_name):
    image_path = os.path.join(IMAGE_DIR, f"{crop_name.lower()}.jpg")

    col1, col2 = st.columns([1.5, 2.5])
    with col1:
        st.markdown(f"<h3 style='color: green;'>🌿 {crop_name.capitalize()} Guide</h3>", unsafe_allow_html=True)
        details = crop_details.get(crop_name.lower(), {})
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
        if os.path.exists(image_path):
            st.image(Image.open(image_path).resize((400, 300)), caption=f"🌿 Recommended Crop: {crop_name}")

# ✅ Streamlit UI
def main():
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🌾 SMART CROP RECOMMENDATION 🌾</h1>", unsafe_allow_html=True)
    
    if st.sidebar.button("🌿 Predict Crop"):
        prediction = predict_crop(50, 50, 50, 25, 50, 6.5, 200)  # Example Inputs
        st.success(f"🌱 Recommended Crop: **{prediction.capitalize()}**")
        show_crop_info(prediction)

if __name__ == '__main__':
    main()
