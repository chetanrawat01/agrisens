import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# ✅ Load dataset
df = pd.read_csv("CROP-RECOMMENDATION\Crop_recommendation.csv")

# ✅ Check unique crops in dataset
print("Unique Crops in Dataset:", df["label"].unique())

# ✅ Define features and target variable
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']

# ✅ Encode target labels (convert crop names into numbers)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# ✅ Train-test split (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ✅ Train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=5)
model.fit(X_train, y_train)

# ✅ Save trained model
with open("RF.pkl", "wb") as f:
    pickle.dump(model, f)

# ✅ Save label encoder
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("✅ Model training complete and saved as RF.pkl!")

# ✅ Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"📊 Model Accuracy: {accuracy * 100:.2f}%")
print("📌 Classification Report:\n", classification_report(y_test, y_pred))

import pandas as pd
import pickle

# ✅ Load trained model
with open("RF.pkl", "rb") as f:
    model = pickle.load(f)

# ✅ Load label encoder
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# ✅ Define feature names (must match training features)
feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

# ✅ Example input (Nitrogen=120, Phosphorus=50, etc.)
input_data = pd.DataFrame([[120, 50, 200, 30, 65, 6.5, 150]], columns=feature_names)

# ✅ Predict crop
predicted_crop = model.predict(input_data)[0]
predicted_crop_name = label_encoder.inverse_transform([predicted_crop])[0]

print(f"🌾 Recommended Crop: {predicted_crop_name}")
