import torch
from model import ServerFailureModel

# Créer notre modèle
model = ServerFailureModel()

# Simuler un batch de 5 serveurs
X = torch.randn(5, 4)

# Effectuer une propagation avant
output = model(X)

print("Architecture du modèle :")
print(model)

print("\nDimensions des entrées :", X.shape)

print("Dimensions des sorties :", output.shape)

# Transformer les logits en probabilités
probabilities = torch.sigmoid(output)

print("\nProbabilités prédites :")
print(probabilities)