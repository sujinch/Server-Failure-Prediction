# Server Failure Prediction — PyTorch

## Présentation

Ce projet est un prototype de classification binaire développé avec PyTorch. Son objectif est d'estimer le risque de panne d'un serveur à partir de quatre indicateurs : utilisation CPU, utilisation RAM, utilisation GPU et température.

Le modèle est accessible via une API REST développée avec FastAPI.

**Important :** le projet utilise des données synthétiques. Il constitue une démonstration technique et non un système validé de prédiction de pannes réelles.

## Technologies utilisées

- Python
- PyTorch
- NumPy et Pandas
- Scikit-learn
- FastAPI et Uvicorn
- Joblib

## Fonctionnement

1. Génération de 5 000 observations synthétiques.
2. Séparation des données en entraînement (70 %), validation (15 %) et test (15 %).
3. Standardisation des caractéristiques à partir du jeu d'entraînement.
4. Construction d'un réseau de neurones entièrement connecté.
5. Entraînement avec Adam et BCEWithLogitsLoss.
6. Sélection du meilleur modèle selon la perte de validation.
7. Évaluation avec accuracy, precision, recall, F1-score et matrice de confusion.
8. Exposition du modèle via FastAPI.

## Architecture du modèle

Le réseau de neurones possède :

- 4 caractéristiques en entrée ;
- une première couche cachée de 16 neurones avec ReLU ;
- une deuxième couche cachée de 8 neurones avec ReLU ;
- une couche de sortie produisant un logit.

La fonction Sigmoid est appliquée pendant l'inférence pour obtenir une probabilité estimée.

## Résultats expérimentaux

| Métrique | Résultat |
|---|---:|
| Accuracy | 74,67 % |
| Precision (classe panne) | 65,45 % |
| Recall (classe panne) | 60,53 % |
| F1-score (classe panne) | 62,89 % |

Ces résultats proviennent d'un jeu de données synthétique. Le jeu de test ayant également été consulté pendant les expérimentations, ils doivent être considérés comme exploratoires plutôt que comme une estimation finale totalement indépendante.

## Installation et exécution

Créer et activer un environnement virtuel Python, puis installer les dépendances :

```bash
python -m pip install torch numpy pandas scikit-learn matplotlib fastapi uvicorn joblib
```

Générer les données :

```bash
python generate_data.py
```

Préparer les données :

```bash
python prepare_data.py
```

Entraîner le réseau de neurones :

```bash
python train.py
```

Évaluer le modèle :

```bash
python evaluate.py
```

Démarrer l'API :

```bash
python -m uvicorn api:app --reload
```

Accéder à la documentation interactive :

`http://127.0.0.1:8000/docs`

## Exemple de requête

**POST /predict**

```json
{
  "cpu_usage": 85,
  "ram_usage": 90,
  "gpu_usage": 95,
  "temperature": 82
}
```

Exemple de réponse obtenue :

```json
{
  "failure_probability": 0.9782,
  "prediction": 1,
  "message": "Risque de panne détecté"
}
```

## Limites et améliorations futures

- Utiliser des données réelles de surveillance des serveurs.
- Effectuer une nouvelle évaluation sur un jeu de test indépendant.
- Choisir le seuil de classification sur le jeu de validation.
- Comparer le réseau de neurones à un modèle de référence simple.
- Tester l'entraînement par mini-batches avec DataLoader.
- Ajouter des tests automatisés, la conteneurisation Docker et du monitoring.
- Étudier la calibration des probabilités prédites.

## Objectif pédagogique

Ce projet m'a permis de mettre en pratique le prétraitement des données, la conception d'un réseau de neurones, la rétropropagation, l'optimisation, l'évaluation des modèles et l'exposition d'un modèle via une API REST.