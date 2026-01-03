import os
import numpy as np
import joblib
from flask import Flask, render_template, request

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load trained model
model = joblib.load(os.path.join(BASE_DIR, "models", "irrigation_model.pkl"))

# 🔴 IMPORTANT: feature order MUST match training
feature_columns = [
    "soil_ph",
    "soil_moisture",
    "nitrogen",
    "phosphorus",
    "potassium",
    "temperature",
    "humidity",
    "rainfall"   # extra feature used during training
]

def generate_explanation(inputs, water):
    reasons = []

    if inputs["soil_moisture"] < 40:
        reasons.append("Low soil moisture increases irrigation requirement")

    if inputs["temperature"] > 30:
        reasons.append("High temperature causes more water loss due to evaporation")

    if inputs["humidity"] < 50:
        reasons.append("Low humidity results in faster soil drying")

    if not reasons:
        reasons.append("Moderate soil and climate conditions require balanced irrigation")

    reasons.append(f"Estimated irrigation requirement is about {round(water, 2)} mm")

    return reasons[:3]

@app.route("/", methods=["GET", "POST"])
def index():
    irrigation = None
    explanation = None
    max_water = 300  # reference for graph

    if request.method == "POST":
        # Inputs from UI
        user_input = {
            "soil_ph": float(request.form["soil_ph"]),
            "soil_moisture": float(request.form["soil_moisture"]),
            "nitrogen": float(request.form["nitrogen"]),
            "phosphorus": float(request.form["phosphorus"]),
            "potassium": float(request.form["potassium"]),
            "temperature": float(request.form["temperature"]),
            "humidity": float(request.form["humidity"]),
            "rainfall": 0.0   # default value (not provided in UI)
        }

        # Arrange input exactly as model expects
        X = np.array([[user_input[col] for col in feature_columns]])

        irrigation = round(float(model.predict(X)[0]), 2)
        explanation = generate_explanation(user_input, irrigation)

    return render_template(
        "index.html",
        irrigation=irrigation,
        explanation=explanation,
        max_water=max_water
    )

if __name__ == "__main__":
    app.run(debug=True)
