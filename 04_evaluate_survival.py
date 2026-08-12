import pandas as pd
from pathlib import Path

# Para NCT00079274

DATASET_NAME = "NCT00079274"

REAL_PATH = Path("data/NCT00079274_unido CSV.csv")
SYN_PATH = Path(f"outputs/{DATASET_NAME}_sintetico_survival_gan.csv")
OUTPUT_DIR = Path("outputs")

TARGET_COLUMN = "dfsstat5"
TIME_COLUMN = "dfstime5"

# Para NCT1150045  

#DATASET_NAME = "NCT1150045"

#REAL_PATH = Path("data/NCT1150045_unido CSV.csv")
#SYN_PATH = Path(f"outputs/{DATASET_NAME}_sintetico_survival_gan.csv")
#OUTPUT_DIR = Path("outputs")

#TARGET_COLUMN = "dfs_stat"
#TIME_COLUMN = "dfs_time"

real = pd.read_csv(REAL_PATH)
syn = pd.read_csv(SYN_PATH)

real = real.dropna(subset=[TARGET_COLUMN, TIME_COLUMN])
real = real[real[TIME_COLUMN] > 0]

# Función para volver la columna numérica a 0/1

def convertir_evento_a_numero(serie):
    return (
        serie.astype(str)
        .str.extract(r"(\d+)")[0]
        .astype(float)
    )

summary = []

for name, df in [("real", real), ("survival_gan", syn)]:
    summary.append({
        "dataset": name,
        "n_rows": len(df),
        "event_rate": convertir_evento_a_numero(df[TARGET_COLUMN]).mean(),
        "time_mean": df[TIME_COLUMN].mean(),
        "time_median": df[TIME_COLUMN].median(),
        "time_min": df[TIME_COLUMN].min(),
        "time_max": df[TIME_COLUMN].max(), 
    })

summary_df = pd.DataFrame(summary)

print(summary_df)

output_path = OUTPUT_DIR / f"{DATASET_NAME}_survival_summary.csv"
summary_df.to_csv(output_path, index=False)

print("Resumen survival guardado:", output_path)