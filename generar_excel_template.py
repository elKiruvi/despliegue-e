import pandas as pd
import numpy as np
import joblib

# ─────────────────────────────────────────────
# Cargar artefactos
# ─────────────────────────────────────────────
scaler = joblib.load('scaler.joblib')
cols_escalar = joblib.load('columnas_escalar.joblib')
cols_categoricas = joblib.load('columnas_categoricas.joblib')
le_binarios = joblib.load('label_encoders_binarios.joblib')
ohe = joblib.load('one_hot_encoder.joblib')
feature_columns = joblib.load('feature_columns.joblib')

# ─────────────────────────────────────────────
# Determinar binarias que son features
# ─────────────────────────────────────────────
binarias_feature = [col for col in le_binarios.keys() if col in feature_columns]

# ─────────────────────────────────────────────
# Generar filas de ejemplo
# ─────────────────────────────────────────────
NUM_FILAS = 10  # Cambia este número si quieres más filas
np.random.seed(42)

datos = {}

# --- Variables numéricas ---
for col in cols_escalar:
    if 'año' in col.lower() or 'anio' in col.lower():
        datos[col] = np.random.randint(2000, 2024, size=NUM_FILAS)
    elif 'integrantes' in col.lower() or 'total' in col.lower():
        datos[col] = np.random.randint(1, 10, size=NUM_FILAS)
    elif 'edad' in col.lower():
        datos[col] = np.random.randint(18, 65, size=NUM_FILAS)
    else:
        datos[col] = np.round(np.random.uniform(0, 100, size=NUM_FILAS), 2)

# --- Variables binarias ---
for col in binarias_feature:
    le = le_binarios[col]
    clases = list(le.classes_)
    datos[col] = np.random.choice(clases, size=NUM_FILAS)

# --- Variables categóricas múltiples ---
for i, col in enumerate(cols_categoricas):
    categorias = list(ohe.categories_[i])
    datos[col] = np.random.choice(categorias, size=NUM_FILAS)

# ─────────────────────────────────────────────
# Crear DataFrame y exportar
# ─────────────────────────────────────────────
df = pd.DataFrame(datos)

# Ordenar columnas: numéricas -> binarias -> categóricas
orden = list(cols_escalar) + binarias_feature + list(cols_categoricas)
# Incluir solo las que existen
orden_final = [c for c in orden if c in df.columns]
df = df[orden_final]

# ─────────────────────────────────────────────
# Guardar Excel
# ─────────────────────────────────────────────
nombre_archivo = 'datos_prediccion.xlsx'
df.to_excel(nombre_archivo, index=False, sheet_name='Datos')

print(f"\n✅ Archivo '{nombre_archivo}' generado con {NUM_FILAS} filas de ejemplo")
print(f"   Columnas: {len(df.columns)}")
print(f"\n📋 Columnas generadas:")
for i, col in enumerate(df.columns, 1):
    print(f"   {i}. {col}")

print(f"\n📊 Vista previa:")
print(df.head().to_string())

# ─────────────────────────────────────────────
# Crear también hoja con valores válidos (referencia)
# ─────────────────────────────────────────────
nombre_referencia = 'referencia_valores_validos.xlsx'

with pd.ExcelWriter(nombre_referencia, engine='openpyxl') as writer:

    # Hoja 1: Resumen de columnas
    resumen = []
    for col in cols_escalar:
        resumen.append({
            'Columna': col,
            'Tipo': 'Numérica',
            'Valores válidos': 'Cualquier número'
        })
    for col in binarias_feature:
        clases = list(le_binarios[col].classes_)
        resumen.append({
            'Columna': col,
            'Tipo': 'Binaria',
            'Valores válidos': ' | '.join(str(c) for c in clases)
        })
    for i, col in enumerate(cols_categoricas):
        cats = list(ohe.categories_[i])
        resumen.append({
            'Columna': col,
            'Tipo': 'Categórica',
            'Valores válidos': ' | '.join(str(c) for c in cats)
        })

    df_resumen = pd.DataFrame(resumen)
    df_resumen.to_excel(writer, sheet_name='Referencia', index=False)

    # Hoja 2: Valores detallados por categórica
    for i, col in enumerate(cols_categoricas):
        cats = list(ohe.categories_[i])
        df_cats = pd.DataFrame({col: cats})
        # Limpiar nombre para hoja (máx 31 chars)
        sheet_name = col[:31].replace('/', '_').replace('\\', '_')
        df_cats.to_excel(writer, sheet_name=sheet_name, index=False)

    for col in binarias_feature:
        clases = list(le_binarios[col].classes_)
        df_bin = pd.DataFrame({col: clases})
        sheet_name = col[:31].replace('/', '_').replace('\\', '_')
        df_bin.to_excel(writer, sheet_name=sheet_name, index=False)

print(f"\n✅ Archivo de referencia '{nombre_referencia}' generado")
print("   Contiene los valores válidos para cada columna categórica")