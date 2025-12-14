# src/test_inference.py
import joblib, pandas as pd
pipe_clf = joblib.load('models/crop_classifier.joblib')
pipe_reg = joblib.load('models/irrigation_regressor.joblib')

row = pd.DataFrame([{
    'temp': 26.0, 'humidity': 50.0, 'light': 12000,
    'vpd': 0.6, 'soil_moisture': 36.0, 'hour': 9, 'soil_depl_3': -0.2
}])
print("Crop prediction:", pipe_clf.predict(row)[0])
print("Irrigation (ml) prediction:", float(pipe_reg.predict(row)[0]))
