import random
import numpy as np
import pandas as pd
from pathlib import Path

from synthcity.benchmark import Benchmarks
from synthcity.plugins.core.dataloader import GenericDataLoader

# 1. CONFIGURACIÓN

SEED = 42

# Descomentar el que se necesite

# Para NCT00079274
DATASET_NAME = "NCT00079274"
DATA_PATH = Path("data/NCT00079274_unido CSV.csv")
TARGET_COLUMN = "dfsstat5"

# Para NCT00079274
#DATASET_NAME = "NCT01150045"
#DATA_PATH = Path("data/NCT01150045_unido CSV.csv")
#TARGET_COLUMN = "dfs_stat"

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# 2. SEMILLAS

random.seed(SEED)
np.random.seed(SEED)

# 3. CARGAR DATASET

df = pd.read_csv(DATA_PATH)

print("Dataset:", DATASET_NAME)
print("Dimensiones originales:", df.shape)

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

print("Dimensiones para benchmark:", df_model.shape)
print("Target:", TARGET_COLUMN)

if TARGET_COLUMN not in df_model.columns:
    raise ValueError(f"La columna target {TARGET_COLUMN} no existe en el dataset")

# 5. DATALOADER CON TARGET PARA TSTR

loader = GenericDataLoader(
    df_model,
    target_column=TARGET_COLUMN,
    random_state=SEED,
)

n_pacientes = len(df_model)

# 6. EXPERIMENTOS

experiments = [
    (
        "tvae_dp_eps3",
        "tvae",
        {
            "random_state": SEED,
            "dp_enabled": True,
            "dp_epsilon": 3.0,
            "dp_delta": 1 / len(df_model),
            "dp_max_grad_norm": 1.0,
        },
    ),
]

# 7. BENCHMARK TSTR

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
        "performance": [
            "linear_model",
            "mlp",
            "xgb",
        ],
    },
)

print(score)

# 8. GUARDAR RESULTADOS

for model_name, model_score in score.items():
    path = OUTPUT_DIR / f"{DATASET_NAME}_benchmark_TSTR_{model_name}.csv"
    model_score.to_csv(path)
    print("Benchmark TSTR guardado:", path)
