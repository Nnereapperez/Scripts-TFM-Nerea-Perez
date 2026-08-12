import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import ks_2samp
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.neighbors import NearestNeighbors

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

DATASETS = [
    {
        "name": "NCT00079274",
        "real": "data/NCT00079274_unido CSV.csv",
        "syn": "outputs/NCT00079274_sintetico_TRexCART.csv"
    },
    {
        "name": "NCT01150045",
        "real": "data/NCT01150045_unido CSV.csv",
        "syn": "outputs/NCT01150045_sintetico_TRexCART.csv"
    }
]


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
        enc = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )
        encoded[cat_cols] = enc.fit_transform(
            encoded[cat_cols].astype(str)
        )

    scaler = StandardScaler()
    encoded[:] = scaler.fit_transform(encoded)

    real_enc = encoded.iloc[:len(real)].to_numpy()
    syn_enc = encoded.iloc[len(real):].to_numpy()

    return real_enc, syn_enc


def ks_test_promedio(real, syn):

    resultados = []

    for col in real.columns:
        if pd.api.types.is_numeric_dtype(real[col]):
            stat, p = ks_2samp(real[col], syn[col])

            resultados.append({
                "variable": col,
                "ks_stat": stat,
                "pvalue": p
            })

    ks_df = pd.DataFrame(resultados)

    return ks_df, float(ks_df["pvalue"].mean())


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


todos_resultados = []

for cfg in DATASETS:

    print("\n================================================")
    print(cfg["name"])
    print("================================================")

    real_df = pd.read_csv(cfg["real"])
    syn_df = pd.read_csv(cfg["syn"])

    print("Real:", real_df.shape)
    print("Sintético:", syn_df.shape)

    real, syn = preparar_datos(real_df, syn_df)

    real_enc, syn_enc = codificar(real, syn)

    ks_df, ks_mean = ks_test_promedio(real, syn)

    alpha = alpha_precision_proxy(real_enc, syn_enc)

    delta = delta_presence_proxy(real_enc, syn_enc)

    resumen = pd.DataFrame([{
        "dataset": cfg["name"],
        "modelo": "TRex-CART",
        "real_rows": len(real),
        "synthetic_rows": len(syn),
        "real_cols": real.shape[1],
        "synthetic_cols": syn.shape[1],
        "ks_test_mean_pvalue": ks_mean,
        "alpha_precision_proxy": alpha,
        "delta_presence_proxy": delta
    }])

    resumen.to_csv(
        OUTPUT_DIR /
        f"{cfg['name']}_benchmark_TRexCART.csv",
        index=False
    )

    ks_df.to_csv(
        OUTPUT_DIR /
        f"{cfg['name']}_benchmark_TRexCART_KS_by_column.csv",
        index=False
    )

    todos_resultados.append(resumen)

    print(resumen)

pd.concat(todos_resultados).to_csv(
    OUTPUT_DIR /
    "benchmark_TRexCART_RESUMEN.csv",
    index=False
)

print("\nBenchmark TRex-CART finalizado")
