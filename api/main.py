# api/main.py
# SenSante API - Assistant pré-diagnostic médical
# Lab 3 - Intégration de Modèles IA - ESP/UCAD

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os
from dotenv import load_dotenv
from groq import Groq
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware

# =========================
# Application FastAPI & CORS
# =========================
app = FastAPI(
    title="SenSante API",
    description="Assistant pré-diagnostic médical pour le Sénégal",
    version="0.2.0"
)

origins = [
    "http://localhost:3000",      # Votre frontend local
    "http://127.0.0.1:3000", 
    "http://localhost:5500"       # Variante courante (Live Server)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           
    allow_credentials=True,
    allow_methods=["*"],              # Autorise POST, GET, OPTIONS, etc.
    allow_headers=["*"],
)   

# Charger les variables d'environnement
load_dotenv()

# Client Groq (charge au démarrage)
groq_client = None
groq_api_key = os.getenv("GROQ_API_KEY")

if groq_api_key:
    groq_client = Groq(api_key=groq_api_key)
    print("Client Groq initialise .")
else:
    print("ATTENTION : GROQ_API_KEY non trouvee . /explain sera desactive .")


# =========================
# Schémas Pydantic
# =========================
class ExplainInput(BaseModel):
    diagnostic: str = Field(..., description="Diagnostic predit par le modele")
    probabilite: float = Field(..., description="Probabilite du diagnostic")
    age: int = Field(...)
    sexe: str = Field(...)
    temperature: float = Field(...)
    region: str = Field(...)

class ExplainOutput(BaseModel):
    explication: str = Field(..., description="Explication en francais")
    modele_llm: str = Field(
        default="llama-3.1-8b-instant",
        description="Modele LLM utilise"
    )

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
# Configuration Prompt LLM
# =========================
SYSTEM_PROMPT = """Tu es un assistant medical senegalais .
Tu recois un diagnostic et des donnees patient .
Explique le resultat en francais simple ,
comme un medecin parlerait a son patient .
Sois rassurant mais recommande toujours
une consultation medicale .
Maximum 3 phrases .
Ne fais JAMAIS de diagnostic toi - meme .
Tu expliques uniquement le diagnostic fourni ."""


# =========================
# Chargement du modèle
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "notebooks" / "models"

print("Chargement du modèle...")

model = joblib.load(MODEL_DIR / "model.pkl")
le_sexe = joblib.load(MODEL_DIR / "encoder_sexe.pkl")
le_region = joblib.load(MODEL_DIR / "encoder_region.pkl")
feature_cols = joblib.load(MODEL_DIR / "feature_cols.pkl")

print(f"Modèle chargé : {list(model.classes_)}")


# =========================
# Routes API
# =========================
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "message": "SenSante API is running"
    }


@app.post("/predict", response_model=DiagnosticOutput)
def predict(patient: PatientInput):
    # Encoder le sexe
    try:
        sexe_enc = le_sexe.transform([patient.sexe])[0]
    except ValueError:
        return DiagnosticOutput(
            diagnostic="erreur",
            probabilite=0.0,
            confiance="aucune",
            message=f"Sexe invalide : {patient.sexe}"
        )

    # Encoder la région
    try:
        region_enc = le_region.transform([patient.region])[0]
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

    proba_max = float(model.predict_proba(features)[0].max())
   
    # Récupérer toutes les probabilités pour chaque classe
    toutes_les_probas = model.predict_proba(features)[0]
    proba_max = float(toutes_les_probas.max())

    # --- AJOUTE CES DEUX LIGNES POUR INSPECTER ---
    print("\n" + "="*40)
    print(f"CLASSES DU MODELE : {model.classes_.tolist()}")
    print(f"PROBABILITES BRUTES : {toutes_les_probas.tolist()}")
    print(f"DIAGNOSTIC : {diagnostic} | PROBA MAX : {proba_max:.2f}")
    print("="*40 + "\n")
    
    
    # Niveau de confiance
    if proba_max >= 0.7:
        confiance = "haute"
    elif proba_max >= 0.4:
        confiance = "moyenne"
    else:
        confiance = "faible"

    # Messages associés
    messages = {
        "palu": "Suspicion de paludisme. Consultez rapidement.",
        "grippe": "Suspicion de grippe. Repos et hydratation.",
        "typh": "Suspicion de typhoïde. Consultation nécessaire.",
        "sain": "Pas de pathologie détectée."
    }

    # Retour API
    return DiagnosticOutput(
        diagnostic=diagnostic,
        probabilite=round(proba_max, 2),
        confiance=confiance,
        message=messages.get(diagnostic, "Consultez un médecin.")
    )


@app.post("/explain", response_model=ExplainOutput)
def explain(data: ExplainInput):
    """Expliquer un diagnostic en francais avec un LLM."""
    if not groq_client:
        return ExplainOutput(
            explication="Service d'explication indisponible . Cle API non configuree .",
            modele_llm="aucun"
        )
    
    # Construire le user prompt
    user_prompt = (
        f"Patient : {data.sexe} , {data.age} ans , "
        f"region {data.region}\n"
        f"Temperature : {data.temperature} C\n"
        f"Diagnostic du modele : {data.diagnostic} "
        f"(probabilite {data.probabilite:.0%}) \n"
        f"Explique ce resultat au patient ."
    )
    
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=200,
            temperature=0.3
        )
        explication = response.choices[0].message.content
    except Exception as e:
        explication = f"Erreur lors de l'appel au LLM : {str(e)}"
        
    return ExplainOutput(explication=explication)


@app.get("/model-info")
def model_info():
    return {
        "model_type": type(model).__name__,
        "n_estimators": getattr(model, "n_estimators", "N/A"),
        "classes": model.classes_.tolist() if hasattr(model, "classes") else None,
        "n_features": model.n_features_in_ if hasattr(model, "n_features_in_") else None
    }