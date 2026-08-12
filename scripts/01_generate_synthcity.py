import random
import numpy as np 
import pandas as pd
from pathlib import Path
import torch

from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader

# 1. CONFIGURACIÓN

SEED = 42

# Columnas deseadas -- Descomentar el que se necesite

# Para NCT00079274
DATASET_NAME = "NCT00079274"
DATA_PATH = Path("data/NCT00079274_unido CSV.csv")

# Para NCT01150045  
#DATASET_NAME = "NCT01150045"
#DATA_PATH = Path("data/NCT01150045_unido CSV.csv")

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

MODELOS = ["tvae", "ddpm"]

# 2. SEMILLAS

random.seed(SEED)
np.random.seed(SEED)

# ---- Dispositivo usado

device = "GPU" if torch.cuda.is_available() else "CPU"
print("Dispositivo usado:", device)

# 3. CARGAR DATASET

df = pd.read_csv(DATA_PATH)

print("Dataset cargado:", DATASET_NAME)
print("Dimensiones reales:", df.shape)

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

# 5. ENTRENAMIENTO Y GENERACIÓN

for nombre_modelo in MODELOS:
    print("Entrenando:", nombre_modelo)

    model = Plugins().get(
        nombre_modelo,
        random_state=SEED,
    )

    model.fit(loader)

    print("PARÁMETROS DEL MODELO:")
    
    params_usados = {
        "dataset": DATASET_NAME,
        "modelo": nombre_modelo,
        "random_state": SEED,
        "loader": "GenericDataLoader",
        "synthetic_size": n_pacientes,
        "device": device,
        "model_type": str(type(model)),
        "model_internal_dict": str(model.__dict__),
    }

    params_path = OUTPUT_DIR / "model_params" / f"{DATASET_NAME}_{nombre_modelo}_params.txt"

    with open(params_path, "w") as f:
        for k, v in params_usados.items():
            f.write(f"{k}: {v}\n")

    print("Parámetros guardados en:", params_path)

    print("Entrenamiento terminado:", nombre_modelo)

    synthetic = model.generate(
        count=n_pacientes,
        random_state=SEED,
    ).dataframe()

    synthetic.insert(0, id_col, [f"SYN_{nombre_modelo.upper()}_{i+1}" for i in range(len(synthetic))])

    output_path = OUTPUT_DIR / f"{DATASET_NAME}_sintetico_{nombre_modelo}.csv"
    synthetic.to_csv(output_path, index=False)

    print("Archivo generado:", output_path)

print("\nProceso terminado.")
