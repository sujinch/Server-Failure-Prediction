import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from model import ServerFailureModel
from prepare_data import X_test_tensor, y_test_tensor


# Créer le modèle
model = ServerFailureModel()

# Charger les paramètres appris
model.load_state_dict(
    torch.load(
        "data/best_model.pth",
        map_location="cpu",
        weights_only=True
    )
)

# Passer le modèle en mode évaluation
model.eval()

# Faire les prédictions sans calculer les gradients
with torch.no_grad():

    # Obtenir les logits
    logits = model(X_test_tensor)

    # Transformer les logits en probabilités
    probabilities = torch.sigmoid(logits)

    # Convertir les probabilités en classes
    predictions = (probabilities >= 0.5).int()


# Convertir les tenseurs en tableaux NumPy
y_true = y_test_tensor.numpy().ravel()
y_pred = predictions.numpy().ravel()

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

print("\n===== ÉVALUATION DU MODÈLE =====")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")

print("\nMatrice de confusion :")
print(confusion_matrix(y_true, y_pred))

print("\nRapport de classification :")
print(classification_report(y_true, y_pred, zero_division=0))