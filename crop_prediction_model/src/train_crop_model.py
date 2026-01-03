import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# -------------------------------------------------
# Load ORIGINAL crop recommendation dataset
# -------------------------------------------------
df = pd.read_csv("data/raw/Crop_recommendation.csv")

# Standardize column names
df = df.rename(columns={
    "N": "nitrogen",
    "P": "phosphorus",
    "K": "potassium",
    "ph": "soil_ph",
    "rainfall": "soil_moisture",
    "label": "crop"
})

# Feature columns
FEATURES = [
    "soil_ph",
    "soil_moisture",
    "nitrogen",
    "phosphorus",
    "potassium",
    "temperature",
    "humidity"
]

X = df[FEATURES]
y = df["crop"]

# -------------------------------------------------
# Encode target labels
# -------------------------------------------------
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# -------------------------------------------------
# Train-test split
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# -------------------------------------------------
# Train Random Forest model
# -------------------------------------------------
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# -------------------------------------------------
# Evaluation
# -------------------------------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# -------------------------------------------------
# GRAPH 1: CONFUSION MATRIX (FIXED – NO BLANK FIGURE)
# -------------------------------------------------
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(8, 8))
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=encoder.classes_
)
disp.plot(ax=ax, cmap="Blues", xticks_rotation=90)

ax.set_title("Confusion Matrix - Crop Prediction")
plt.tight_layout()
plt.show()

# -------------------------------------------------
# GRAPH 2: FEATURE IMPORTANCE
# -------------------------------------------------
importances = model.feature_importances_

plt.figure(figsize=(8, 5))
plt.barh(FEATURES, importances)
plt.xlabel("Importance")
plt.title("Feature Importance - Crop Prediction Model")
plt.tight_layout()
plt.show()

# -------------------------------------------------
# Save model & encoder
# -------------------------------------------------
joblib.dump(model, "crop_model.pkl")
joblib.dump(encoder, "crop_encoder.pkl")

print("crop_model.pkl and crop_encoder.pkl saved successfully")
