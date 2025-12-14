# src/train_models.py
"""
Minimal trainer for greenhouse demo.
Run: python src/train_models.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error, r2_score
import joblib, math

DATA_PATH = Path('data/greenhouse.csv')
MODELS_DIR = Path('models')
MODELS_DIR.mkdir(exist_ok=True)

def compute_vpd(temp_c, rh_pct):
    svp = 0.6108 * np.exp((17.27 * temp_c) / (temp_c + 237.3))
    avp = svp * (rh_pct / 100.0)
    return svp - avp

if not DATA_PATH.exists():
    print("No data/greenhouse.csv found. Create one (I'll show you a sample in Step 2). Exiting.")
    raise SystemExit(1)

df = pd.read_csv(DATA_PATH, parse_dates=['timestamp']).sort_values('timestamp').reset_index(drop=True)
df[['temp','humidity','light','soil_moisture']] = df[['temp','humidity','light','soil_moisture']].apply(pd.to_numeric, errors='coerce')
df = df.fillna(method='ffill').fillna(0)

df['vpd'] = compute_vpd(df['temp'], df['humidity'])
df['hour'] = df['timestamp'].dt.hour
df['soil_depl_3'] = df['soil_moisture'] - df['soil_moisture'].shift(3)
df['soil_depl_3'] = df['soil_depl_3'].fillna(0)

# create irrigation_ml if missing (very simple synthetic rule)
if 'irrigation_ml' not in df.columns:
    df['irrigation_ml'] = ((60 - df['soil_moisture']).clip(lower=0) * 10).astype(float).clip(lower=0)

if 'crop_label' not in df.columns:
    df['crop_label'] = 'unknown'

features = ['temp','humidity','light','vpd','soil_moisture','hour','soil_depl_3']

# classifier
X = df[features]
y = df['crop_label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
clf_pipe = Pipeline([('scale', StandardScaler()), ('clf', RandomForestClassifier(n_estimators=100, random_state=42))])
clf_pipe.fit(X_train, y_train)
y_pred = clf_pipe.predict(X_test)
print("Classifier report:\n", classification_report(y_test, y_pred))
joblib.dump(clf_pipe, MODELS_DIR / 'crop_classifier.joblib')
print("Saved crop classifier.")

# regressor
Xr = df[features]
yr = df['irrigation_ml']
Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, test_size=0.2, random_state=42)
reg_pipe = Pipeline([('scale', StandardScaler()), ('reg', RandomForestRegressor(n_estimators=100, random_state=42))])
reg_pipe.fit(Xr_train, yr_train)
yr_pred = reg_pipe.predict(Xr_test)
rmse = math.sqrt(mean_squared_error(yr_test, yr_pred))
print(f"Regressor RMSE: {rmse:.2f} ml, R2: {r2_score(yr_test, yr_pred):.3f}")
joblib.dump(reg_pipe, MODELS_DIR / 'irrigation_regressor.joblib')
print("Saved irrigation regressor.")
