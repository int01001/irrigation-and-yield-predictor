import os
import joblib
from flask import Flask, render_template, request

app = Flask(__name__)
BASE = os.path.dirname(os.path.abspath(__file__))

# ===== LOAD MODELS =====
crop_model = joblib.load(os.path.join(BASE, "models/crop_model.pkl"))
crop_target_encoder = joblib.load(os.path.join(BASE, "models/crop_target_encoder.pkl"))

irrigation_model = joblib.load(os.path.join(BASE, "models/irrigation_model.pkl"))
feature_columns = joblib.load(os.path.join(BASE, "models/feature_columns.pkl"))

yield_model = joblib.load(os.path.join(BASE, "models/yield_model.pkl"))
crop_input_encoder = joblib.load(os.path.join(BASE, "models/crop_input_encoder.pkl"))

@app.route("/")
def home():
    return render_template("crop.html")

# ===== CROP PREDICTION =====
@app.route("/crop", methods=["GET", "POST"])
def crop():
    prediction = None
    soil_data = None
    explanation = None

    if request.method == "POST":
        soil_data = [
            float(request.form["soil_ph"]),
            float(request.form["soil_moisture"]),
            float(request.form["nitrogen"]),
            float(request.form["phosphorus"]),
            float(request.form["potassium"]),
            float(request.form["temperature"]),
            float(request.form["humidity"])
        ]

        pred = crop_model.predict([soil_data])[0]
        prediction = crop_target_encoder.inverse_transform([pred])[0]

        explanation = (
            f"The model analyzed soil nutrients, moisture and climate. "
            f"Based on these conditions, **{prediction}** provides optimal growth "
            f"and nutrient efficiency."
        )

    return render_template(
        "crop.html",
        prediction=prediction,
        soil_data=soil_data,
        explanation=explanation
    )

# ===== IRRIGATION =====
@app.route("/irrigation", methods=["GET", "POST"])
def irrigation():
    irrigation = None
    explanation = None

    if request.method == "POST":
        inputs = {
            "soil_ph": float(request.form["soil_ph"]),
            "soil_moisture": float(request.form["soil_moisture"]),
            "nitrogen": float(request.form["nitrogen"]),
            "phosphorus": float(request.form["phosphorus"]),
            "potassium": float(request.form["potassium"]),
            "temperature": float(request.form["temperature"]),
            "humidity": float(request.form["humidity"]),
            "rainfall": float(request.form["rainfall"]),
        }

        ordered = [inputs[c] for c in feature_columns]
        irrigation = round(irrigation_model.predict([ordered])[0], 2)

        explanation = (
            "The irrigation requirement is calculated by analyzing soil moisture, "
            "rainfall, temperature, and nutrient levels to avoid over- or under-watering."
        )

    return render_template(
        "irrigation.html",
        irrigation=irrigation,
        explanation=explanation
    )

# ===== YIELD =====
@app.route("/yield", methods=["GET", "POST"])
def yield_page():
    prediction = None
    explanation = None
    crops = list(crop_input_encoder.classes_)

    if request.method == "POST":
        crop = request.form["crop"]
        encoded_crop = crop_input_encoder.transform([crop])[0]

        features = [
            float(request.form["area"]),
            float(request.form["rainfall"]),
            float(request.form["fertilizer"]),
            float(request.form["pesticide"]),
            encoded_crop
        ]

        prediction = round(yield_model.predict([features])[0], 2)

        explanation = (
            "Yield is predicted using historical production patterns, "
            "area cultivated, rainfall, and input usage for the selected crop."
        )

    return render_template(
        "yield.html",
        prediction=prediction,
        crops=crops,
        explanation=explanation
    )

if __name__ == "__main__":
    app.run(debug=True)
