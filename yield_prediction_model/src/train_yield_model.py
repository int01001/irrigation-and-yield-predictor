import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# =========================
# 1. LOAD DATASET
# =========================
df = pd.read_csv("data/raw/crop_yield.csv")

# =========================
# 2. ENCODE CROP (CRITICAL)
# =========================
crop_encoder = LabelEncoder()
df["crop_encoded"] = crop_encoder.fit_transform(df["Crop"])

# =========================
# 3. SELECT FEATURES
# =========================
features = [
    "Area",
    "Annual_Rainfall",
    "Fertilizer",
    "Pesticide",
    "crop_encoded"
]

X = df[features]
y = df["Yield"]

# =========================
# 4. TRAIN / TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 5. TRAIN MODEL
# =========================
model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# =========================
# 6. EVALUATION
# =========================
preds = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE:", rmse)

# =========================
# 7. SAVE ARTIFACTS
# =========================
joblib.dump(model, "models/yield_model.pkl")
joblib.dump(crop_encoder, "models/crop_encoder.pkl")

print("Saved: yield_model.pkl and crop_encoder.pkl")

# =========================
# 8. VISUALIZATION
# =========================
plt.figure(figsize=(6,4))
plt.scatter(y_test, preds, alpha=0.6, color="#ff9800")
plt.xlabel("Actual Yield")
plt.ylabel("Predicted Yield")
plt.title("Yield Prediction Performance")
plt.tight_layout()
plt.show()
