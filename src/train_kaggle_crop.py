# src/train_kaggle_crop.py
"""
Train a crop-recommendation classifier from Crop_recommendation.csv
Saves pipeline to models/crop_recommender.joblib

Run:
    python src/train_kaggle_crop.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import argparse

# ---------- Config ----------
DATA_PATH = Path("data/Crop_recommendation.csv")   # file you uploaded
OUT_MODEL = Path("models/crop_recommendercls.joblib")
TEST_SIZE = 0.2
RANDOM_STATE = 42

def load_data(path):
    df = pd.read_csv(path)
    return df

def prepare_X_y(df):
    # Expected numeric features in this dataset
    features = ['N','P','K','temperature','humidity','ph','rainfall']
    for f in features:
        if f not in df.columns:
            raise KeyError(f"Expected feature column '{f}' not found in CSV")
    X = df[features].astype(float)
    y = df['label'].astype(str)
    return X, y

def build_pipeline():
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1))
    ])
    return pipe

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=str(DATA_PATH), help="Path to CSV")
    parser.add_argument("--out", type=str, default=str(OUT_MODEL), help="Output model path")
    args = parser.parse_args()

    df = load_data(args.data)
    print("Loaded data. Rows:", len(df))
    X, y = prepare_X_y(df)

    # encode labels for convenience in some displays (pipeline handles raw labels)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    class_names = list(le.classes_)
    print("Classes:", class_names)

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_enc)
    print(f"Train rows: {len(X_train)}, Test rows: {len(X_test)}")

    # build and fit pipeline
    pipeline = build_pipeline()
    print("Training RandomForest classifier...")
    pipeline.fit(X_train, y_train)

    # evaluate
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {acc:.4f}\n")
    print("Classification report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # cross-validated accuracy (optional quick check)
    print("Cross-val accuracy (5-fold on full dataset, may take a little time):")
    try:
        cv_scores = cross_val_score(pipeline, X, y_enc, cv=5, scoring='accuracy', n_jobs=-1)
        print("CV accuracies:", np.round(cv_scores, 4))
        print("CV mean:", float(np.mean(cv_scores)))
    except Exception as e:
        print("CV error (continuing):", e)

    # Save the pipeline and label encoder together
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {"pipeline": pipeline, "label_encoder": le, "feature_names": X.columns.tolist()}
    joblib.dump(bundle, out_path)
    print(f"Saved model bundle to {out_path}")

    # Print top feature importances
    try:
        clf = pipeline.named_steps["clf"]
        importances = clf.feature_importances_
        for name, imp in sorted(zip(X.columns, importances), key=lambda x: x[1], reverse=True):
            print(f"{name}: {imp:.4f}")
    except Exception:
        pass

    print("Done.")

if __name__ == "__main__":
    main()
