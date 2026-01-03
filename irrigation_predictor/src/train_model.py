
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Load dataset
df = pd.read_csv("data/raw/irrigation_synthetic.csv")

# Encode crop column
encoder = LabelEncoder()
df["crop"] = encoder.fit_transform(df["crop"])

# Features and target
X = df.drop("irrigation_mm", axis=1)
y = df["irrigation_mm"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(
    n_estimators=150,
    random_state=42
)
model.fit(X_train, y_train)

# Predictions
preds = model.predict(X_test)

# RMSE (version-safe)
rmse = sqrt(mean_squared_error(y_test, preds))

# Save model and encoder
joblib.dump(model, "models/irrigation_model.pkl")
joblib.dump(encoder, "models/crop_encoder.pkl")

print("Model trained successfully")
print(f"RMSE: {rmse:.2f}")

# ---------- Visualization ----------
plt.figure(figsize=(8, 5))
plt.scatter(y_test, preds, alpha=0.6)
plt.xlabel("Actual Irrigation (mm)")
plt.ylabel("Predicted Irrigation (mm)")
plt.title("Actual vs Predicted Irrigation Requirement")
plt.grid(True)
plt.tight_layout()
plt.show()
