import random
import numpy as np
import pandas as pd
from pathlib import Path
import torch

from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import SurvivalAnalysisDataLoader

# 1. CONFIGURACIÓN

SEED = 42

# Columnas deseadas -- Descomentar el que se necesite

# Para NCT00079274

#DATASET_NAME = "NCT00079274"
#DATA_PATH = Path("data/NCT00079274_unido CSV.csv")

#TARGET_COLUMN = "dfsstat5"
#TIME_COLUMN = "dfstime5"

# Para NCT01150045  

DATASET_NAME = "NCT01150045"
DATA_PATH = Path("data/NCT01150045_unido CSV.csv")

TARGET_COLUMN = "dfs_stat"
TIME_COLUMN = "dfs_time"

OUTPUT_DIR = Path("outputs")
(OUTPUT_DIR / "model_params").mkdir(exist_ok=True)

# 2. SEMILLAS

random.seed(SEED)
np.random.seed(SEED)

# ---- Dispositivo usado

device = "GPU" if torch.cuda.is_available() else "CPU"
print("Dispositivo usado:", device)

# 3. CARGAR DATASET

df = pd.read_csv(DATA_PATH)

print("Dataset cargado:", df.shape)


id_col = df.columns[0]
print("Columna ID:", id_col)

df = df.dropna(subset=[TARGET_COLUMN, TIME_COLUMN])

# 4. LIMPIAR
# SurvivalGAN exige tiempo > 0
df = df[df[TIME_COLUMN] > 0]

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())

print("Dataset survival limpio:", df.shape)

loader = SurvivalAnalysisDataLoader(
    df,
    target_column=TARGET_COLUMN,
    time_to_event_column=TIME_COLUMN,
    random_state=SEED,
)

# 5. ENTRENAMIENTO Y GENERACIÓN

model = Plugins().get(
        "survival_gan",
        random_state=SEED,
    )

print("Entrenando SurvivalGAN...")
    
model.fit(loader)


print("PARÁMETROS DEL MODELO:")
    
params_usados = {
    "dataset": DATASET_NAME,
    "modelo": "survival_gan",
    "target_column": TARGET_COLUMN,
    "time_column": TIME_COLUMN,
    "random_state": SEED,
    "loader": "SurvivalAnalysisDataLoader",
    "synthetic_size": len(df),
    "device": device,
    "model_type": str(type(model)),
    "model_internal_dict": str(model.__dict__),
}

params_path = OUTPUT_DIR / "model_params" / f"{DATASET_NAME}_survival_gan_params.txt"

with open(params_path, "w") as f:
    for k, v in params_usados.items():
        f.write(f"{k}: {v}\n")

print("Parámetros guardados en:", params_path)
print("Entrenamiento terminado")

synthetic = model.generate(
    count=len(df),
    random_state=SEED,
).dataframe()

if id_col in synthetic.columns:
    synthetic[id_col] = [f"SURV_{i+1}" for i in range (len(synthetic))]
else:
    synthetic.insert(0, id_col, [f"SURV_{i+1}" for i in range(len(synthetic))])

output_path = OUTPUT_DIR / f"{DATASET_NAME}_sintetico_survival_gan.csv"
synthetic.to_csv(output_path, index=False)

print("Archivo generado:", output_path)
