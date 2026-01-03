
import os
import numpy as np
import joblib
from flask import Flask, render_template, request

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "models", "yield_model.pkl"))
crop_encoder = joblib.load(os.path.join(BASE_DIR, "models", "crop_encoder.pkl"))

FEATURES = [
    "Area",
    "Annual_Rainfall",
    "Fertilizer",
    "Pesticide",
    "crop_encoded"
]

def explain(inputs, y):
    reasons = []
    if inputs["Annual_Rainfall"] > 1800:
        reasons.append("High rainfall supports better crop growth")
    if inputs["Fertilizer"] > 500000:
        reasons.append("High fertilizer usage improves nutrient availability")
    if inputs["Area"] > 5000:
        reasons.append("Larger cultivated area stabilizes yield estimation")
    reasons.append(f"Predicted yield is approximately {y} tons/hectare")
    return reasons[:3]

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    explanation = None

    if request.method == "POST":
        crop = request.form["crop"]

        data = {
            "Area": float(request.form["area"]),
            "Annual_Rainfall": float(request.form["rainfall"]),
            "Fertilizer": float(request.form["fertilizer"]),
            "Pesticide": float(request.form["pesticide"]),
            "crop_encoded": crop_encoder.transform([crop])[0]
        }

        X = np.array([[data[col] for col in FEATURES]])
        prediction = round(float(model.predict(X)[0]), 2)
        explanation = explain(data, prediction)

    return render_template(
        "index.html",
        prediction=prediction,
        explanation=explanation,
        crops=list(crop_encoder.classes_)
    )

if __name__ == "__main__":
    app.run(debug=True)
