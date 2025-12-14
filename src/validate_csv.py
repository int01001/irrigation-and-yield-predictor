# src/validate_csv.py
import pandas as pd
from pathlib import Path
p = Path('data/greenhouse.csv')
if not p.exists():
    raise SystemExit("data/greenhouse.csv not found.")
df = pd.read_csv(p, parse_dates=['timestamp'])
expected = ['timestamp','temp','humidity','light','soil_moisture','crop_label','irrigation_ml']
missing = [c for c in expected if c not in df.columns]
if missing:
    print("Missing columns:", missing)
else:
    print("All expected columns present.")
print("Row count:", len(df))
print(df.head().to_string(index=False))
