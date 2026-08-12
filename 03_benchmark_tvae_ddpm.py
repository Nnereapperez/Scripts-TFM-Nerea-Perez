import random
import numpy as np
import pandas as pd
from pathlib import Path

from synthcity.benchmark import Benchmarks
from synthcity.plugins.core.dataloader import GenericDataLoader

# 1. CONFIGURACIÓN

SEED = 42

DATASET_NAME = "NCT00079274"
DATA_PATH = Path("data/NCT00079274_unido CSV.csv")

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# 2. SEMILLAS

random.seed(SEED)
np.random.seed(SEED)

# 3. CARGAR DATASET

df = pd.read_csv(DATA_PATH)

id_col = df.columns[0]
print("Columna ID:", id_col)

df_model = df.drop(columns=[id_col])
df_model = df_model.dropna(axis=1, how="all")

# 4. LIMPIAR

for col in df_model.columns:
    if df_model[col].dtype == "object":
        df_model[col] = df_model[col].fillna(df_model[col].mode()[0])
    else:
        df_model[col] = df_model[col].fillna(df_model[col].median())

print("Dataset para modelar:", df_model.shape)

loader = GenericDataLoader(df_model, random_state=SEED)
n_pacientes = len(df_model)

experiments = [
    ("tvae_benchmark", "tvae", {
        "random_state": SEED,
    }),
    ("ddpm_benchmark_light", "ddpm", {
        "random_state": SEED,
        "n_iter": 100,
    }),
]

score = Benchmarks.evaluate(
    experiments,
    loader,
    #synthetic_size=n_pacientes
    synthetic_size=min(n_pacientes, 1000),
    repeats=1,
    metrics={
        "stats": [
            "alpha_precision",
            "ks_test",
            "max_mean_discrepancy",
        ],
        "privacy": [
            "delta-presence",
        ],
    },
)

print(score)

for model_name, model_score in score.items():
    path = OUTPUT_DIR / f"{DATASET_NAME}_benchmark_{model_name}.csv"
    model_score.to_csv(path)
    print("Benchmark guardado:", path)