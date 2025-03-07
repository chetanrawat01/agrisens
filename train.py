import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# ✅ Define Paths
CSV_PATH = "CROP-RECOMMENDATION/Crop_recommendation.csv"
MODEL_PATH = "CROP-RECOMMENDATION/RF.pkl"
ENCODER_PATH = "CROP-RECOMMENDATION/label_encoder.pkl"

# ✅ Load dataset
df = pd.read_csv(CSV_PATH)

# ✅ Define features and labels
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']

# ✅ Encode labels (convert crop names into numbers)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# ✅ Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ✅ Train the model
model = RandomForestClassifier(n_estimators=100, random_state=5)
model.fit(X_train, y_train)

# ✅ Save the trained model
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

# ✅ Save the label encoder
with open(ENCODER_PATH, "wb") as f:
    pickle.dump(label_encoder, f)

print("✅ Model training complete and saved as RF.pkl & label_encoder.pkl!")
