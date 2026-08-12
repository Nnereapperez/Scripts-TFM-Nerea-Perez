import pandas as pd
import numpy as np
from scipy.stats import iqr
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder

# 1. CONFIGURACIÓN - descomentar el dataset a usar

# --- NCT00079274 ---
# INPUT_FILE   = "data/NCT00079274_unido CSV.csv"
# OUTPUT_FILE  = "outputs/NCT00079274_synthetic_CART_sklearn_1to1.csv"
# N_PATIENTS   = 2686
# DATASET_ID   = "NCT00079274"

# --- NCT01150045 --- (descomentar y comentar el bloque de arriba)
INPUT_FILE  = "data/NCT01150045_unido CSV.csv"
OUTPUT_FILE = "outputs/NCT01150045_synthetic_CART_sklearn_1to1.csv"
N_PATIENTS  = 2527
DATASET_ID  = "NCT01150045"

SMOOTHING    = True   # añade variabilidad continua a variables numéricas
PROPER       = True   # resampling durante entrenamiento, reduce sobreajuste
MINIBUCKET   = 5
RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)

# 2. CARGA E IMPUTACIÓN

df      = pd.read_csv(INPUT_FILE)
id_col  = df.columns[0]
df_real = df.drop(columns=[id_col]).copy()
print(f"{DATASET_ID} — {df_real.shape[0]} pacientes, {df_real.shape[1]} variables")

df_work  = df_real.copy()
num_cols = df_work.select_dtypes(include='number').columns
cat_cols = df_work.select_dtypes(exclude='number').columns
df_work[num_cols] = df_work[num_cols].fillna(df_work[num_cols].median())
df_work[cat_cols] = df_work[cat_cols].fillna(df_work[cat_cols].mode().iloc[0])

# 3. CODIFICACIÓN — LabelEncoder para todas las categóricas

encoders   = {}
df_encoded = df_work.copy()
for col in cat_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_work[col].astype(str))
    encoders[col] = le

# 4. ENTRENAMIENTO — un árbol por columna

print("Entrenando CART...")
columns   = list(df_encoded.columns)
models    = {}
leaf_vals = {}
y_bounds  = {}

for col in columns:
    X = df_encoded.drop(columns=[col]).values
    y = df_encoded[col].values

    if PROPER:
        idx = np.random.choice(len(X), size=len(X), replace=True)
        X_train, y_train = X[idx], y[idx]
    else:
        X_train, y_train = X, y

    if col in cat_cols:
        model = DecisionTreeClassifier(min_samples_leaf=MINIBUCKET, random_state=RANDOM_STATE)
    else:
        model = DecisionTreeRegressor(min_samples_leaf=MINIBUCKET, random_state=RANDOM_STATE)
        y_bounds[col] = (float(np.min(y)), float(np.max(y)))

    model.fit(X_train, y_train)
    models[col] = model

    leaves    = model.apply(X)
    leaf_dict = {}
    for leaf_id, val in zip(leaves, y):
        leaf_dict.setdefault(leaf_id, []).append(val)
    leaf_vals[col] = {k: np.array(v) for k, v in leaf_dict.items()}

print("Entrenamiento completado")

# 5. GENERACIÓN

print(f"Generando {N_PATIENTS} pacientes sintéticos...")
sample_size   = max(N_PATIENTS, len(df_encoded))
synthetic_enc = df_encoded.sample(n=sample_size, replace=True, random_state=RANDOM_STATE).reset_index(drop=True)
predictions   = {}

for col in columns:
    model       = models[col]
    X_test      = synthetic_enc.drop(columns=[col]).values
    leaves_pred = model.apply(X_test)
    y_pred      = np.empty(len(leaves_pred), dtype=object)

    for i, leaf_id in enumerate(leaves_pred):
        if leaf_id in leaf_vals[col]:
            y_pred[i] = np.random.choice(leaf_vals[col][leaf_id])
        else:
            y_pred[i] = model.predict(X_test[i:i+1])[0]

    y_pred = np.array(y_pred, dtype=float if col in num_cols else object)

    if SMOOTHING and col in num_cols:
        y_min, y_max = y_bounds[col]
        idx = (y_pred != y_min) & (y_pred != y_max)
        if idx.sum() >= 2:
            std_val = np.std(y_pred[idx].astype(float))
            iqr_val = iqr(y_pred[idx].astype(float)) / 1.34
            if std_val > 0 or iqr_val > 0:
                bw = 0.9 * idx.sum() ** (-1/5) * np.minimum(std_val, iqr_val)
                y_pred[idx] = np.array([np.random.normal(v, bw) for v in y_pred[idx].astype(float)])

    predictions[col] = y_pred

synthetic_df = pd.DataFrame(predictions).iloc[:N_PATIENTS].reset_index(drop=True)

# 6. DECODIFICACIÓN

for col in cat_cols:
    le    = encoders[col]
    codes = synthetic_df[col].astype(float).round().astype(int).clip(0, len(le.classes_) - 1)
    synthetic_df[col] = le.inverse_transform(codes)

for col in num_cols:
    synthetic_df[col] = synthetic_df[col].astype(float)

# 7. VERIFICACIÓN Y GUARDADO

collapsed = [c for c in synthetic_df.columns if synthetic_df[c].nunique() == 1]
print(f"NaN: {synthetic_df.isna().sum().sum()} | Colapsadas: {len(collapsed)}")

for col in ['arm', 'sex', 'agecat', 'racecat']:
    if col in df_real.columns and col in synthetic_df.columns:
        r = df_real[col].value_counts(normalize=True)
        s = synthetic_df[col].value_counts(normalize=True)
        print(f"  {col}: diff máx = {max(abs(s.get(v,0)-r.get(v,0)) for v in r.index):.3f}")

for col in ['bmi2', 'dfstime5', 'futime8', 'os_time', 'dfs_time']:
    if col in df_real.columns and col in synthetic_df.columns:
        print(f"  {col}: real {df_real[col].mean():.1f}±{df_real[col].std():.1f} | sint {synthetic_df[col].mean():.1f}±{synthetic_df[col].std():.1f}")

synthetic_df.insert(0, id_col, [f"SYN_CART_{i+1}" for i in range(N_PATIENTS)])
synthetic_df.to_csv(OUTPUT_FILE, index=False)
print(f"\nGuardado: {OUTPUT_FILE} — {synthetic_df.shape}")