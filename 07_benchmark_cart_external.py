import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import ks_2samp
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.neighbors import NearestNeighbors

# 1. CONFIGURACIÓN - Descomentar el dataset a usar

# Para NCT00079274

DATASET_NAME = "NCT00079274"
REAL_PATH = Path("data/NCT00079274_unido CSV.csv")
SYN_PATH = Path("outputs/NCT00079274_synthetic_CART_sklearn_1to1.csv")

# Para NCT01150045

#DATASET_NAME = "NCT01150045"
#REAL_PATH = Path("data/NNCT01150045_unido CSV.csv")
#SYN_PATH = Path("outputs/NCT01150045_synthetic_CART_sklearn_1to1.csv")

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# 2. FUNCIONES

def preparar_datos(real_df, syn_df):
    id_col = real_df.columns[0]

    real = real_df.drop(columns=[id_col], errors="ignore").copy()
    syn = syn_df.drop(columns=[id_col], errors="ignore").copy()

    common_cols = [c for c in real.columns if c in syn.columns]

    real = real[common_cols]
    syn = syn[common_cols]

    for col in common_cols:
        if real[col].dtype == "object":
            valor = real[col].mode()[0]
            real[col] = real[col].fillna(valor)
            syn[col] = syn[col].fillna(valor)
        else:
            valor = real[col].median()
            real[col] = real[col].fillna(valor)
            syn[col] = syn[col].fillna(valor)

    return real, syn


def codificar(real, syn):
    combined = pd.concat([real, syn], axis=0)

    cat_cols = combined.select_dtypes(exclude="number").columns.tolist()

    encoded = combined.copy()

    if cat_cols:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        encoded[cat_cols] = enc.fit_transform(encoded[cat_cols].astype(str))

    scaler = StandardScaler()
    encoded[:] = scaler.fit_transform(encoded)

    real_enc = encoded.iloc[:len(real)].to_numpy()
    syn_enc = encoded.iloc[len(real):].to_numpy()

    return real_enc, syn_enc


def ks_test_promedio(real, syn):
    valores = []

    for col in real.columns:
        if pd.api.types.is_numeric_dtype(real[col]):
            stat, pvalue = ks_2samp(real[col], syn[col])
            valores.append(pvalue)

    if len(valores) == 0:
        return np.nan

    return float(np.mean(valores))


def alpha_precision_proxy(real_enc, syn_enc):
    centro = real_enc.mean(axis=0)

    dist_real = np.linalg.norm(real_enc - centro, axis=1)
    dist_syn = np.linalg.norm(syn_enc - centro, axis=1)

    umbral = np.percentile(dist_real, 95)

    return float(np.mean(dist_syn <= umbral))


def delta_presence_proxy(real_enc, syn_enc):
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(syn_enc)

    distancias, _ = nn.kneighbors(real_enc)

    umbral = np.percentile(distancias, 5)

    return float(np.mean(distancias <= umbral))

# 3. CARGA

real_df = pd.read_csv(REAL_PATH)
syn_df = pd.read_csv(SYN_PATH)

print("Real:", real_df.shape)
print("Sintético CART:", syn_df.shape)

real, syn = preparar_datos(real_df, syn_df)

real_enc, syn_enc = codificar(real, syn)

# 4. MÉTRICAS

resultados = {
    "dataset": DATASET_NAME,
    "modelo": "TRex-CART",
    "real_rows": len(real),
    "synthetic_rows": len(syn),
    "real_cols": real.shape[1],
    "synthetic_cols": syn.shape[1],
    "ks_test_mean_pvalue": ks_test_promedio(real, syn),
    "alpha_precision_proxy": alpha_precision_proxy(real_enc, syn_enc),
    "delta_presence_proxy": delta_presence_proxy(real_enc, syn_enc),
}

resultados_df = pd.DataFrame([resultados])

print(resultados_df)

# 5. GUARDAR

output_path = OUTPUT_DIR / f"{DATASET_NAME}_benchmark_cart_external.csv"
resultados_df.to_csv(output_path, index=False)

print("Benchmark CART guardado en:", output_path)
