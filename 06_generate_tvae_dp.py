import random
import numpy as np 
import pandas as pd
from pathlib import Path
import torch

from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader

# 1. CONFIGURACIÓN

SEED = 42

DATASET_NAME = "NCT00079274"
DATA_PATH = Path("data/NCT00079274_unido CSV.csv")

OUTPUT_DIR = Path("outputs")
(OUTPUT_DIR / "model_params").mkdir(exist_ok=True)

EPSILONS = [0.1, 1.0, 3.0, 10.0]

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

# Delta recomendado: 1/n
delta = 1 / n_pacientes

print(f"N pacientes: {n_pacientes}")
print(f"Delta: {delta:.8f}")

# 5. TVAE CON PRIVACIDAD DIFERENCIAL (dp)

resultados = []

for epsilon in EPSILONS:
    print("\n" + "=" * 60)
    print(f"Entrenando TVAE-DP | epsilon={epsilon} | delta={delta:.8f}")
    print("=" * 60)

    try:
        model = Plugins().get(
            "tvae",
            dp_enabled=True,
            dp_epsilon=epsilon,
            dp_delta=delta,
            dp_max_grad_norm=1.0,
            random_state=SEED,
        )

        model.fit(loader)

        print("PARÁMETROS DEL MODELO:")
    
        params_usados = {
            "dataset": DATASET_NAME,
            "modelo": "tvae_dp",
            "epsilon": epsilon,
            "delta": delta,
            "dp_enabled": True,
            "dp_max_grad_norm": 1.0,
            "random_state": SEED,
            "loader": "GenericDataLoader",
            "synthetic_size": n_pacientes,
            "device": device,
            "model_type": str(type(model)),
            "model_internal_dict": str(model.__dict__),
        }

        params_path = OUTPUT_DIR / "model_params" / f"{DATASET_NAME}_tvae_dp_epsilon_{epsilon}_params.txt"

        with open(params_path, "w") as f:
            for k, v in params_usados.items():
                f.write(f"{k}: {v}\n")

        print("Parámetros guardados en:", params_path)
        print("Entrenamiento terminado")

        synthetic = model.generate(
            count=n_pacientes,
            random_state=SEED,
        ).dataframe()

        synthetic.insert(
            0,
            id_col,
            [f"SYN_TVAE_DP_eps{epsilon}_{i+1}" for i in range(len(synthetic))]
        )

        output_path = OUTPUT_DIR / f"{DATASET_NAME}_sintetico_tvae_dp_epsilon{epsilon}.csv"
        synthetic.to_csv(output_path, index=False)

        print("Archivo generado:", output_path)
        print("Shape:", synthetic.shape)

        resultados.append({
            "epsilon": epsilon,
            "delta": delta,
            "estado": "OK",
            "archivo": str(output_path),
        })

    except Exception as e:
        print(f"ERROR con epsilon={epsilon}")
        print(e)

        resultados.append({
            "epsilon": epsilon,
            "delta": delta,
            "estado": "ERROR",
            "archivo": str(e)
        })

# 6. RESUMEN

resumen = pd.DataFrame(resultados)

resumen_path = OUTPUT_DIR / f"{DATASET_NAME}_resumen_tvae_dp.csv"
resumen.to_csv(resumen_path, index=False)

print("\nResumen:")
print(resumen)

print("Resumen guardado en:", resumen_path)