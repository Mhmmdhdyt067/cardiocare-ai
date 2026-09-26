from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

app = FastAPI(title="CardioCare ML API")

# Load Model & Scaler saat server booting
try:
    model = joblib.load("heart_disease_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_cols = joblib.load("feature_columns.pkl")
except Exception as e:
    print(f"Error loading models: {e}")

# Schema Input Pasien dari Laravel
class PatientInput(BaseModel):
    age: int
    sex: str          # 'M' atau 'F'
    chestPainType: str # 'TA', 'ATA', 'NAP', 'ASY'
    restingBP: int
    cholesterol: int
    fastingBS: int     # 0 atau 1
    restingECG: str   # 'Normal', 'ST', 'LVH'
    maxHR: int
    exerciseAngina: str # 'Y' atau 'N'
    oldpeak: float
    stSlope: str      # 'Up', 'Flat', 'Down'

@app.get("/")
def read_root():
    return {"status": "ok", "message": "CardioCare ML API is running!"}

@app.post("/predict")
def predict(data: PatientInput):
    try:
        # Auto-imputasi Kolesterol 0 ke median 237 jika user memasukkan 0
        eff_cholesterol = 237 if data.cholesterol <= 0 else data.cholesterol

        # Format data input menjadi DataFrame
        raw_df = pd.DataFrame([{
            'Age': data.age,
            'Sex': data.sex,
            'ChestPainType': data.chestPainType,
            'RestingBP': data.restingBP,
            'Cholesterol': eff_cholesterol,
            'FastingBS': data.fastingBS,
            'RestingECG': data.restingECG,
            'MaxHR': data.maxHR,
            'ExerciseAngina': data.exerciseAngina,
            'Oldpeak': data.oldpeak,
            'ST_Slope': data.stSlope
        }])

        # One-Hot Encoding
        encoded_df = pd.get_dummies(raw_df, dtype=int)

        # Samakan struktur kolom dengan data pelatihan
        full_df = pd.DataFrame(0, index=[0], columns=feature_cols)
        for col in encoded_df.columns:
            if col in full_df.columns:
                full_df[col] = encoded_df[col]

        # Feature Scaling & Prediksi
        scaled_input = scaler.transform(full_df)
        prediction = int(model.predict(scaled_input)[0])
        probability = float(model.predict_proba(scaled_input)[0][1]) * 100

        return {
            "status": "success",
            "prediction": prediction, # 0 = Normal, 1 = Berisiko Jantung
            "label": "Berisiko Penyakit Jantung" if prediction == 1 else "Normal / Rendah Risiko",
            "probability": round(probability, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
