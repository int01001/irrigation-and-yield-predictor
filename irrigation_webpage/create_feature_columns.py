import joblib

feature_columns = [
    "soil_ph",
    "soil_moisture",
    "nitrogen",
    "phosphorus",
    "potassium",
    "temperature",
    "humidity",
    "rainfall"   # or whatever extra column your model used
]

joblib.dump(feature_columns, "feature_columns.pkl")
print("feature_columns.pkl created")
