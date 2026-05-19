# api/main.py
# SenSante API - Assistant pré-diagnostic médical
# Lab 3 - Intégration de Modèles IA - ESP/UCAD

from fastapi import FastAPI
from pydantic import BaseModel, Field

import joblib
import numpy as np

from pathlib import Path


# =========================
# Schémas Pydantic
# =========================

class PatientInput(BaseModel):
    age: int = Field(..., ge=0, le=120)
    sexe: str = Field(...)
    temperature: float = Field(..., ge=35.0, le=42.0)
    tension_sys: int = Field(..., ge=60, le=250)
    toux: bool = Field(...)
    fatigue: bool = Field(...)
    maux_tete: bool = Field(...)
    region: str = Field(...)


class DiagnosticOutput(BaseModel):
    diagnostic: str
    probabilite: float
    confiance: str
    message: str


# =========================
# Application FastAPI
# =========================

app = FastAPI(
    title="SenSante API",
    description="Assistant pré-diagnostic médical pour le Sénégal",
    version="0.2.0"
)

from fastapi.middleware.cors import CORSMiddleware

# Autoriser les requetes depuis le frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En dev : tout accepter
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# =========================
# Chargement du modèle
# =========================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "notebooks" / "models"

print("Chargement du modèle...")

model = joblib.load(MODEL_DIR / "model.pkl")

le_sexe = joblib.load(
    MODEL_DIR / "encoder_sexe.pkl"
)

le_region = joblib.load(
    MODEL_DIR / "encoder_region.pkl"
)

feature_cols = joblib.load(
    MODEL_DIR / "feature_cols.pkl"
)

print(f"Modèle chargé : {list(model.classes_)}")


# =========================
# Routes
# =========================

@app.get("/health")
def health_check():

    return {
        "status": "ok",
        "message": "SenSante API is running"
    }


@app.post(
    "/predict",
    response_model=DiagnosticOutput
)
def predict(patient: PatientInput):

    # Encoder le sexe
    try:
        sexe_enc = le_sexe.transform(
            [patient.sexe]
        )[0]

    except ValueError:

        return DiagnosticOutput(
            diagnostic="erreur",
            probabilite=0.0,
            confiance="aucune",
            message=f"Sexe invalide : {patient.sexe}"
        )

    # Encoder la région
    try:
        region_enc = le_region.transform(
            [patient.region]
        )[0]

    except ValueError:

        return DiagnosticOutput(
            diagnostic="erreur",
            probabilite=0.0,
            confiance="aucune",
            message=f"Région inconnue : {patient.region}"
        )

    # Construire les features
    features = np.array([[
        patient.age,
        sexe_enc,
        patient.temperature,
        patient.tension_sys,
        int(patient.toux),
        int(patient.fatigue),
        int(patient.maux_tete),
        region_enc
    ]])

    # Prédiction
    diagnostic = model.predict(features)[0]

    proba_max = float(
        model.predict_proba(features)[0].max()
    )

    # Niveau de confiance
    if proba_max >= 0.7:
        confiance = "haute"

    elif proba_max >= 0.4:
        confiance = "moyenne"

    else:
        confiance = "faible"

    # Messages associés
    messages = {
        "palu": (
            "Suspicion de paludisme. "
            "Consultez rapidement."
        ),

        "grippe": (
            "Suspicion de grippe. "
            "Repos et hydratation."
        ),

        "typh": (
            "Suspicion de typhoïde. "
            "Consultation nécessaire."
        ),

        "sain": (
            "Pas de pathologie détectée."
        )
    }

    # Retour API
    return DiagnosticOutput(
        diagnostic=diagnostic,
        probabilite=round(proba_max, 2),
        confiance=confiance,
        message=messages.get(
            diagnostic,
            "Consultez un médecin."
        )
    )

# app = FastAPI()

# # Charger ton modèle (exemple)
# model = joblib.load("notebooks/models/model.pkl")

# @app.get("/model-info")
# def model_info():
#     return {
#         "model_type": type(model).__name__,
#         "n_estimators": getattr(model, "n_estimators", "N/A"),
#         "classes": model.classes_.tolist() if hasattr(model, "classes_") else None,
#         "n_features": model.n_features_in_ if hasattr(model, "n_features_in_") else None
#     }
    
    