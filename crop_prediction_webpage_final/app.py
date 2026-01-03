import os
import numpy as np
import joblib
from flask import Flask, render_template, request

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "models", "crop_model.pkl"))
encoder = joblib.load(os.path.join(BASE_DIR, "models", "crop_encoder.pkl"))

def generate_explanation(features, crop_name):
    soil_ph, moisture, n, p, k, temp, humidity = features

    reasons = []

    if moisture > 100:
        reasons.append("High soil moisture supports water-intensive crops")

    if humidity > 70:
        reasons.append("High humidity favors healthy crop growth")

    if temp >= 24 and temp <= 32:
        reasons.append("Temperature is ideal for tropical crops")

    if soil_ph >= 6 and soil_ph <= 7.5:
        reasons.append("Soil pH is suitable for nutrient absorption")

    if not reasons:
        reasons.append("Overall soil and climate conditions match crop requirements")

    return reasons[:3]  # keep explanation short & clear

@app.route("/", methods=["GET", "POST"])
def index():
    top_crops = None
    explanation = None
    chart_labels = None
    chart_values = None

    if request.method == "POST":
        features = [
            float(request.form["soil_ph"]),
            float(request.form["soil_moisture"]),
            float(request.form["nitrogen"]),
            float(request.form["phosphorus"]),
            float(request.form["potassium"]),
            float(request.form["temperature"]),
            float(request.form["humidity"])
        ]

        X = np.array(features).reshape(1, -1)

        probs = model.predict_proba(X)[0]
        crops = encoder.classes_

        top_idx = np.argsort(probs)[-3:][::-1]

        top_crops = []
        chart_labels = []
        chart_values = []

        for i in top_idx:
            name = crops[i]
            conf = round(probs[i] * 100, 2)
            top_crops.append({"name": name, "confidence": conf})
            chart_labels.append(name)
            chart_values.append(conf)

        # Explanation for best crop
        explanation = generate_explanation(features, top_crops[0]["name"])

    return render_template(
        "index.html",
        top_crops=top_crops,
        explanation=explanation,
        chart_labels=chart_labels,
        chart_values=chart_values
    )

if __name__ == "__main__":
    app.run(debug=True)
