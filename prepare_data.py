import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Charger notre jeu de données
df = pd.read_csv("data/server_data.csv")

# Séparer les caractéristiques et la variable cible
X = df.drop(columns=["failure"])
y = df["failure"]

print("Caractéristiques :", X.shape)
print("Variable cible :", y.shape)

# Séparer 70 % pour l'entraînement et 30 % pour la suite
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# Diviser les 30 % restants en deux parties égales
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("\nRépartition des données :")
print("Training :", len(X_train))
print("Validation :", len(X_val))
print("Test :", len(X_test))

# Créer notre outil de standardisation
scaler = StandardScaler()

# Les statistiques de standardisation doivent être calculées uniquement à partir des données d'entraînement sinon on risque de "fuiter" des informations du test dans l'entraînement.
# Apprendre moyenne et écart-type uniquement sur le training
X_train_scaled = scaler.fit_transform(X_train)

# Appliquer la même transformation aux autres ensembles
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Convertir les caractéristiques en tenseurs float32
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

# Convertir les étiquettes en tenseurs float32
y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.float32).unsqueeze(1)
y_val_tensor = torch.tensor(y_val.to_numpy(), dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test.to_numpy(), dtype=torch.float32).unsqueeze(1)

print("\nDimensions des tenseurs :")
print("X_train :", X_train_tensor.shape)
print("y_train :", y_train_tensor.shape)

import joblib

joblib.dump(scaler, "data/scaler.pkl")