import torch
import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel, Field

from model import ServerFailureModel


# Créer l'application API
app = FastAPI(title="Server Failure Prediction API")

# Charger le modèle entraîné
model = ServerFailureModel()

model.load_state_dict(
    torch.load(
        "data/best_model.pth",
        map_location="cpu",
        weights_only=True
    )
)

model.eval()

# Charger le scaler utilisé pendant l'entraînement
scaler = joblib.load("data/scaler.pkl")


# Définir les données attendues
class ServerData(BaseModel):
    cpu_usage: float = Field(ge=0, le=100)
    ram_usage: float = Field(ge=0, le=100)
    gpu_usage: float = Field(ge=0, le=100)
    temperature: float = Field(ge=0, le=120)


@app.get("/")
def home():
    return {"message": "API de prédiction des pannes opérationnelle"}


@app.post("/predict")
def predict(data: ServerData):

    # Construire un tableau avec les caractéristiques
    features = pd.DataFrame([data.model_dump()])

    # Appliquer la même standardisation
    features_scaled = scaler.transform(features)

    # Convertir en tenseur PyTorch
    input_tensor = torch.tensor(
        features_scaled,
        dtype=torch.float32
    )

    # Effectuer la prédiction
    with torch.no_grad():
        logit = model(input_tensor)
        probability = torch.sigmoid(logit).item()

    # Appliquer le seuil de classification
    prediction = int(probability >= 0.5)

    return {
        "failure_probability": round(probability, 4),
        "prediction": prediction,
        "message": (
            "Risque de panne détecté"
            if prediction == 1
            else "Pas de panne prédite"
        )
    }