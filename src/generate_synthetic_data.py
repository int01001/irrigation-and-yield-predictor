# src/generate_synthetic_data.py
import argparse, pandas as pd, numpy as np
from datetime import datetime, timedelta

parser = argparse.ArgumentParser()
parser.add_argument('--rows', type=int, default=500)
parser.add_argument('--out', type=str, default='data/greenhouse.csv')
args = parser.parse_args()

rows = args.rows
start = datetime(2025,12,1,6,0,0)   # 6 AM
interval = timedelta(minutes=5)     # reading every 5 minutes

timestamps = [start + i*interval for i in range(rows)]
temp = 24 + 3 * np.sin(np.linspace(0, 3.14, rows)) + np.random.normal(0,0.3,rows)
humidity = 60 - 6 * np.sin(np.linspace(0, 3.14, rows)) + np.random.normal(0,1.0,rows)
light = (np.clip( (np.sin(np.linspace(-1.5, 1.5, rows)) + 1) * 20000, 0, 20000 )).astype(int)

soil = np.zeros(rows)
soil[0] = 50.0
for i in range(1, rows):
    soil[i] = soil[i-1] - 0.02 - 0.01 * (temp[i]-25)/2 + np.random.normal(0,0.05)
    if soil[i] < 15:
        soil[i] = 15.0

irrigation = np.zeros(rows)
for i in range(rows):
    if soil[i] < 35 and np.random.rand() < 0.1:
        vol = int((40 - soil[i]) * 12 + np.random.randint(50,150))
        irrigation[i] = vol
        soil[i] += vol/12.0

crop_label = ['tomato'] * rows

df = pd.DataFrame({
    'timestamp': [t.isoformat() for t in timestamps],
    'temp': np.round(temp,2),
    'humidity': np.round(humidity,2),
    'light': light,
    'soil_moisture': np.round(soil,2),
    'crop_label': crop_label,
    'irrigation_ml': irrigation
})
df.to_csv(args.out, index=False)
print("Wrote", args.out, "with", len(df), "rows")
