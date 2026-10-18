import numpy as np
import pandas as pd

# Pour obtenir les mêmes données à chaque exécution
rng = np.random.default_rng(42)

# Nombre d'observations
n_samples = 5000

# Génération des caractéristiques
cpu_usage = rng.uniform(10, 100, n_samples)
ram_usage = rng.uniform(10, 100, n_samples)
gpu_usage = rng.uniform(0, 100, n_samples)

# Température simulée, partiellement liée à la charge CPU et GPU
temperature = (
    30
    + 0.30 * cpu_usage
    + 0.25 * gpu_usage
    + rng.normal(0, 8, n_samples)
)

temperature = np.clip(temperature, 25, 100)

# Score de risque artificiel
risk_score = (
    -7
    + 0.025 * cpu_usage
    + 0.020 * ram_usage
    + 0.020 * gpu_usage
    + 0.045 * temperature
)

# Transformation du score en probabilité
failure_probability = 1 / (1 + np.exp(-risk_score))

# Génération des étiquettes de panne
failure = rng.binomial(1, failure_probability)

# Création du DataFrame
df = pd.DataFrame({
    "cpu_usage": cpu_usage,
    "ram_usage": ram_usage,
    "gpu_usage": gpu_usage,
    "temperature": temperature,
    "failure": failure
})

# Enregistrement dans un fichier CSV
df.to_csv("data/server_data.csv", index=False)

# Affichage des premières observations
print(df.head())

print("\nDimensions du dataset :", df.shape)

print("\nRépartition des classes :")
print(df["failure"].value_counts())