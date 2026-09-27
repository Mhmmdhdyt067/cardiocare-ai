import os
import joblib
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Instance FastAPI wajib bernama 'app'
app = FastAPI()

# Tambahkan CORS Middleware di sini
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Mengizinkan semua origin (termasuk http://localhost:8000 / 3000 / 5173)
    allow_credentials=True,
    allow_methods=["*"],  # Mengizinkan semua method (GET, POST, OPTIONS, dll.)
    allow_headers=["*"],
)

# Path absolut agar file .pkl selalu terdeteksi di Vercel
BASE_DIR = Path(__file__).resolve().parent

try:
    model = joblib.load(BASE_DIR / "heart_disease_model.pkl")
    scaler = joblib.load(BASE_DIR / "scaler.pkl")
    feature_cols = joblib.load(BASE_DIR / "feature_columns.pkl")
except Exception as e:
    print(f"Error loading model files: {e}")

class PatientInput(BaseModel):
    age: int
    sex: str
    chestPainType: str
    restingBP: int
    cholesterol: int
    fastingBS: int
    restingECG: str
    maxHR: int
    exerciseAngina: str
    oldpeak: float
    stSlope: str

@app.get("/")
def read_root():
    return {"status": "ok", "message": "CardioCare ML API on Vercel is live!"}

@app.post("/predict")
def predict(data: PatientInput):
    try:
        eff_cholesterol = 237 if data.cholesterol <= 0 else data.cholesterol

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

        encoded_df = pd.get_dummies(raw_df, dtype=int)
        full_df = pd.DataFrame(0, index=[0], columns=feature_cols)
        
        for col in encoded_df.columns:
            if col in full_df.columns:
                full_df[col] = encoded_df[col]

        scaled_input = scaler.transform(full_df)
        prediction = int(model.predict(scaled_input)[0])
        probability = float(model.predict_proba(scaled_input)[0][1]) * 100

        return {
            "status": "success",
            "prediction": prediction,
            "label": "Berisiko Penyakit Jantung" if prediction == 1 else "Normal / Rendah Risiko",
            "probability": round(probability, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
