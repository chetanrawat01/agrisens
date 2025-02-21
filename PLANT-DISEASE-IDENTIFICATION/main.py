import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ✅ **Get the absolute path**
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "trained_plant_disease_model.keras")
IMAGE_PATH = os.path.join(BASE_DIR, "images", "Diseases.png")

# ✅ **Check if model exists before loading**
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    st.error(f"❌ Error: Model file not found at {MODEL_PATH}")

# ✅ **Model Prediction Function**
def model_prediction(test_image):
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element

# ✅ **Sidebar Navigation**
st.sidebar.title("AgriSens")
app_mode = st.sidebar.selectbox("Select Page", ["HOME", "DISEASE RECOGNITION"])

# ✅ **Check if Disease Image Exists**
if os.path.exists(IMAGE_PATH):
    img = Image.open(IMAGE_PATH)
    st.image(img, caption="Plant Disease Chart", use_column_width=True)
else:
    st.error(f"❌ Error: Disease image not found at {IMAGE_PATH}")

# ✅ **Home Page**
if app_mode == "HOME":
    st.markdown("<h1 style='text-align: center;'>SMART DISEASE DETECTION</h1>", unsafe_allow_html=True)

# ✅ **Disease Recognition Page**
elif app_mode == "DISEASE RECOGNITION":
    st.header("📷 Disease Recognition")

    test_image = st.file_uploader("Choose an Image:", type=["png", "jpg", "jpeg"])

    if test_image:
        st.image(test_image, caption="Uploaded Image", use_column_width=True)

    # Predict button
    if st.button("Predict"):
        if test_image:
            st.snow()
            st.write("🔍 **Analyzing Image...**")
            result_index = model_prediction(test_image)

            # ✅ **Disease Class Labels**
            class_names = [
                "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
                "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew",
                "Cherry_(including_sour)___healthy", "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
                "Corn_(maize)___Common_rust_", "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy",
                "Grape___Black_rot", "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
                "Grape___healthy", "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot",
                "Peach___healthy", "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy",
                "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
                "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew",
                "Strawberry___Leaf_scorch", "Strawberry___healthy", "Tomato___Bacterial_spot",
                "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold",
                "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite",
                "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus",
                "Tomato___healthy"
            ]

            predicted_label = class_names[result_index]
            st.success(f"✅ **Model Prediction:** {predicted_label}")
        else:
            st.error("⚠️ Please upload an image before clicking 'Predict'.")
