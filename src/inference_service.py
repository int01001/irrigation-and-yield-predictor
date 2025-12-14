# src/inference_service.py
"""
Inference service for greenhouse.
Usage:
  # simulate from CSV (fast testing):
  python src/inference_service.py --simulate data/greenhouse.csv --interval 0.1

  # listen to serial (Arduino) and reply:
  python src/inference_service.py --serial COM3 --baud 9600

Expect sensor line format (CSV) from Arduino or simulate file:
timestamp,temp,humidity,light,soil_moisture
e.g.
2025-12-08T12:00:00,26.4,45.2,12000,34.0
"""
import argparse
import time
import json
from datetime import datetime
import joblib
import pandas as pd
import numpy as np

# Optional: import serial only when needed to avoid dependency if only simulating
try:
    import serial
except Exception:
    serial = None

MODEL_CROP = 'models/crop_classifier.joblib'
MODEL_REG = 'models/irrigation_regressor.joblib'

# ---- Config ----
PUMP_FLOW_ML_PER_MIN = 100.0   # adjust to your pump calibration (ml per minute)
MAX_ML = 1000.0
MIN_ML = 0.0

# ---- Helpers ----
def compute_vpd(temp_c, rh_pct):
    svp = 0.6108 * np.exp((17.27 * temp_c) / (temp_c + 237.3))
    avp = svp * (rh_pct / 100.0)
    return svp - avp

def build_feature_row(d):
    # expects dict with keys: timestamp, temp, humidity, light, soil_moisture
    t = d.get('timestamp')
    if isinstance(t, str):
        try:
            hour = datetime.fromisoformat(t).hour
        except Exception:
            # fallback if timezone or formatting differs
            hour = datetime.strptime(t.split('.')[0], "%Y-%m-%dT%H:%M:%S").hour
    elif isinstance(t, datetime):
        hour = t.hour
    else:
        hour = int(d.get('hour', 0))
    temp = float(d.get('temp', 0.0))
    hum = float(d.get('humidity', 0.0))
    light = float(d.get('light', 0.0))
    soil = float(d.get('soil_moisture', 0.0))
    vpd = compute_vpd(temp, hum)
    # soil_depl_3 can't be computed from a single row; set 0 or accept if provided
    soil_depl_3 = float(d.get('soil_depl_3', 0.0))
    row = {
        'temp': temp,
        'humidity': hum,
        'light': light,
        'vpd': vpd,
        'soil_moisture': soil,
        'hour': hour,
        'soil_depl_3': soil_depl_3
    }
    return row

def clamp_ml(x):
    return max(MIN_ML, min(MAX_ML, float(x)))

# ---- Inference logic ----
class InferenceService:
    def __init__(self, model_crop_path=MODEL_CROP, model_reg_path=MODEL_REG):
        print("Loading models...")
        self.clf = None
        self.reg = None
        try:
            m = joblib.load(model_crop_path)
            # training saved pipeline as full pipeline; classifier might be entire pipeline or dict
            self.clf = m
        except Exception as e:
            print("Crop classifier not loaded (continuing):", e)
        try:
            self.reg = joblib.load(model_reg_path)
        except Exception as e:
            raise SystemExit(f"Failed to load regressor: {e}")
        print("Models loaded.")

    def predict(self, feature_row):
        # feature_row is dict; convert to DataFrame with same columns order used in training
        df = pd.DataFrame([feature_row])
        crop = None
        try:
            if self.clf is not None:
                crop = str(self.clf.predict(df)[0])
        except Exception as e:
            print("Classifier predict error:", e)
        ml = 0.0
        try:
            ml = float(self.reg.predict(df)[0])
        except Exception as e:
            print("Regressor predict error:", e)
            ml = 0.0
        ml = clamp_ml(ml)
        # duration in seconds
        duration_min = ml / PUMP_FLOW_ML_PER_MIN
        duration_sec = int(round(duration_min * 60))
        return {'crop': crop, 'ml': float(ml), 'duration_sec': duration_sec}

# ---- Serial helpers ----
def parse_sensor_line(line):
    # line expected: timestamp,temp,humidity,light,soil_moisture
    parts = [p.strip() for p in line.strip().split(',')]
    if len(parts) < 5:
        raise ValueError("Bad line, expected 5 values")
    return {
        'timestamp': parts[0],
        'temp': float(parts[1]),
        'humidity': float(parts[2]),
        'light': float(parts[3]),
        'soil_moisture': float(parts[4])
    }

def run_simulate(csv_path, interval, svc: InferenceService):
    print("Simulating from", csv_path, "interval", interval)
    df = pd.read_csv(csv_path, parse_dates=['timestamp'])
    for _, r in df.iterrows():
        d = {
            'timestamp': r['timestamp'].isoformat(),
            'temp': float(r['temp']),
            'humidity': float(r['humidity']),
            'light': int(r['light']),
            'soil_moisture': float(r['soil_moisture']),
            # allow rows that include soil_depl_3 if present
            'soil_depl_3': float(r.get('soil_depl_3', 0.0)) if 'soil_depl_3' in r else 0.0
        }
        feat = build_feature_row(d)
        out = svc.predict(feat)
        # Print as JSON to stdout so you can observe it
        print(json.dumps({'input': d, 'command': out}))
        time.sleep(interval)

def run_serial(port, baud, svc: InferenceService):
    if serial is None:
        raise SystemExit("pyserial not installed or import failed. Install pyserial to use serial mode.")
    print(f"Opening serial port {port} at {baud} baud...")
    ser = serial.Serial(port, baud, timeout=2)
    print("Serial open. Listening for sensor lines...")
    try:
        while True:
            raw = ser.readline().decode('utf-8').strip()
            if not raw:
                continue
            try:
                sensor = parse_sensor_line(raw)
            except Exception as e:
                print("Failed to parse line:", e, "raw:", raw)
                continue
            feat = build_feature_row(sensor)
            out = svc.predict(feat)
            # send command back as JSON line
            cmd = json.dumps(out)
            ser.write((cmd + '\n').encode('utf-8'))
            print("IN:", raw)
            print("OUT:", cmd)
    except KeyboardInterrupt:
        print("Serial listener stopped by user.")
    finally:
        ser.close()

# ---- CLI ----
def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--simulate', type=str, help='CSV file to simulate sensor feed (must contain timestamp,temp,humidity,light,soil_moisture)')
    group.add_argument('--serial', type=str, help='Serial port name (e.g., COM3 or /dev/ttyUSB0)')
    parser.add_argument('--baud', type=int, default=9600, help='Serial baud rate (default 9600)')
    parser.add_argument('--interval', type=float, default=1.0, help='Interval between simulated rows (seconds)')
    parser.add_argument('--pump_flow', type=float, default=PUMP_FLOW_ML_PER_MIN, help='Pump flow ml/min calibration')
    args = parser.parse_args()

    global PUMP_FLOW_ML_PER_MIN
    PUMP_FLOW_ML_PER_MIN = args.pump_flow

    svc = InferenceService()
    if args.simulate:
        run_simulate(args.simulate, args.interval, svc)
    else:
        run_serial(args.serial, args.baud, svc)

if __name__ == '__main__':
    main()
