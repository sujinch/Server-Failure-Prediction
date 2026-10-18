import torch
import torch.nn as nn
import torch.optim as optim

from model import ServerFailureModel

from prepare_data import (
    X_train_tensor,
    y_train_tensor,
    X_val_tensor,
    y_val_tensor
)

# Pour rendre les résultats reproductibles
torch.manual_seed(42)

# Créer notre réseau de neurones
model = ServerFailureModel()

# Fonction de perte
criterion = nn.BCEWithLogitsLoss()

# Optimiseur
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Nombre d'epochs
epochs = 300

best_val_loss = float("inf")
patience = 10
patience_counter = 0

for epoch in range(epochs):

    # -------- ENTRAÎNEMENT --------

    model.train()

    # 1. Remettre les gradients à zéro
    optimizer.zero_grad()

    # 2. Forward propagation
    outputs = model(X_train_tensor)

    # 3. Calculer la fonction de perte
    loss = criterion(outputs, y_train_tensor)

    # 4. Backpropagation
    loss.backward()

    # 5. Mettre à jour les paramètres
    optimizer.step()

    # -------- VALIDATION --------

    model.eval()

    with torch.no_grad():

        val_outputs = model(X_val_tensor)

        val_loss = criterion(
            val_outputs,
            y_val_tensor
        )

    # Afficher les résultats
    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Training Loss: {loss.item():.4f} | "
        f"Validation Loss: {val_loss.item():.4f}"
    )

    # -------- EARLY STOPPING --------

    if val_loss.item() < best_val_loss:

        best_val_loss = val_loss.item()

        patience_counter = 0

        # Sauvegarder les meilleurs paramètres
        torch.save(
            model.state_dict(),
            "data/best_model.pth"
        )

    else:

        patience_counter += 1

        if patience_counter >= patience:

            print("\nEarly stopping activé !")
            break

print("\nEntraînement terminé !")

