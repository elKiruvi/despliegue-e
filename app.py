import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ─────────────────────────────────────────────
# Configuración
# ─────────────────────────────────────────────
st.set_page_config(page_title="Predictor de Red Neuronal", layout="centered")

@st.cache_resource
def cargar_artefactos():
    modelo = joblib.load('modelo_red_neuronal.joblib')
    scaler = joblib.load('scaler.joblib')
    cols_escalar = joblib.load('columnas_escalar.joblib')
    cols_categoricas = joblib.load('columnas_categoricas.joblib')
    le_binarios = joblib.load('label_encoders_binarios.joblib')
    ohe = joblib.load('one_hot_encoder.joblib')
    feature_columns = joblib.load('feature_columns.joblib')
    return modelo, scaler, cols_escalar, cols_categoricas, le_binarios, ohe, feature_columns

try:
    modelo, scaler, cols_escalar, cols_categoricas, le_binarios, ohe, feature_columns = cargar_artefactos()
except Exception as e:
    st.error(f"❌ Error al cargar artefactos: {e}")
    st.stop()

# Binarias que son features
binarias_feature = [col for col in le_binarios.keys() if col in feature_columns]

# ─────────────────────────────────────────────
# MAPEO AMIGABLE PARA BINARIAS
# ─────────────────────────────────────────────
ETIQUETAS_BINARIAS = {
    0: "No",
    1: "Sí",
    '0': "No",
    '1': "Sí",
    0.0: "No",
    1.0: "Sí"
}

def etiqueta_amigable(valor):
    """Convierte 0/1 en texto legible."""
    return ETIQUETAS_BINARIAS.get(valor, str(valor))

def valor_original(etiqueta, clases):
    """Convierte texto legible de vuelta al valor que espera el encoder."""
    inverso = {etiqueta_amigable(c): c for c in clases}
    return inverso.get(etiqueta, etiqueta)


st.title("🧠 Clasificador - Red Neuronal")

# ─────────────────────────────────────────────
# Función de preprocesamiento
# ─────────────────────────────────────────────
def preprocesar(df_input):
    df = df_input.copy()

    for col in cols_escalar:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df[cols_escalar] = scaler.transform(df[cols_escalar])

    for col in binarias_feature:
        if col in df.columns:
            le = le_binarios[col]
            df[col] = le.transform(df[col].astype(str))

    df_cat = df[cols_categoricas]
    cat_encoded = ohe.transform(df_cat)
    if hasattr(cat_encoded, "toarray"):
        cat_encoded = cat_encoded.toarray()

    try:
        ohe_col_names = ohe.get_feature_names_out(cols_categoricas)
    except AttributeError:
        ohe_col_names = ohe.get_feature_names(cols_categoricas)

    df_cat_encoded = pd.DataFrame(cat_encoded, columns=ohe_col_names, index=df.index)

    df_final = pd.concat([df.drop(columns=cols_categoricas), df_cat_encoded], axis=1)
    df_final = df_final.reindex(columns=feature_columns, fill_value=0)
    df_final = df_final.fillna(0)

    return df_final


# ─────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────
tab1, tab2 = st.tabs(["📝 Entrada Manual", "📂 Cargar Excel"])

# ═══════════════════════════════════════════════
# TAB 1: ENTRADA MANUAL
# ═══════════════════════════════════════════════
with tab1:
    st.markdown("Ingresa los datos manualmente para una sola predicción.")

    with st.form("formulario_manual"):

        st.subheader("📊 Variables Numéricas")
        input_data = {}

        col1, col2 = st.columns(2)
        with col1:
            if 'Año desmovilización' in cols_escalar:
                input_data['Año desmovilización'] = st.number_input(
                    'Año desmovilización',
                    min_value=1950, max_value=2050, value=2010, step=1
                )
        with col2:
            if 'Total Integrantes grupo familiar' in cols_escalar:
                input_data['Total Integrantes grupo familiar'] = st.number_input(
                    'Total Integrantes grupo familiar',
                    min_value=1, max_value=30, value=1, step=1
                )

        for col in cols_escalar:
            if col not in input_data:
                input_data[col] = st.number_input(
                    f"{col}", value=0.0, key=f"m_num_{col}"
                )

        # ─────────────────────────────────────
        # BINARIAS CON ETIQUETAS AMIGABLES ✅
        # ─────────────────────────────────────
        st.subheader("🔘 Variables Binarias")
        selecciones_binarias = {}  # Guardar selección amigable

        for col in binarias_feature:
            le = le_binarios[col]
            clases = list(le.classes_)

            # Crear opciones amigables: "No" / "Sí" en vez de 0 / 1
            opciones_amigables = [etiqueta_amigable(c) for c in clases]

            seleccion = st.selectbox(
                col,
                opciones_amigables,
                key=f"m_bin_{col}"
            )
            selecciones_binarias[col] = seleccion

            # Guardar el VALOR ORIGINAL que espera el encoder
            input_data[col] = valor_original(seleccion, clases)

        st.subheader("📋 Variables Categóricas")
        for i, col in enumerate(cols_categoricas):
            opciones = list(ohe.categories_[i])
            input_data[col] = st.selectbox(col, opciones, key=f"m_cat_{col}")

        btn_manual = st.form_submit_button("🚀 Predecir")

    if btn_manual:
        try:
            df_input = pd.DataFrame([input_data])

            # Mostrar datos con etiquetas amigables
            df_display = df_input.copy()
            for col in binarias_feature:
                if col in df_display.columns:
                    df_display[col] = df_display[col].map(
                        lambda x: etiqueta_amigable(x)
                    )

            st.info("📝 Datos ingresados:")
            st.dataframe(df_display, use_container_width=True)

            df_final = preprocesar(df_input)
            X = df_final.values.astype(float)

            with st.spinner('Procesando...'):
                pred = modelo.predict(X)
                proba = modelo.predict_proba(X)

            st.success("✅ Predicción completada")
            st.markdown(f"### 🎯 Resultado: `{pred[0]}`")

            clases = modelo.classes_
            df_probs = pd.DataFrame({
                "Clase": clases,
                "Probabilidad": [f"{p:.2f}%" for p in proba[0] * 100]
            })
            st.dataframe(df_probs, use_container_width=True)

            st.bar_chart(
                pd.DataFrame(proba[0], index=clases, columns=["Probabilidad"])
            )

        except Exception as e:
            st.error(f"❌ Error: {e}")
            import traceback
            st.code(traceback.format_exc())

# ═══════════════════════════════════════════════
# TAB 2: CARGA DE EXCEL
# ═══════════════════════════════════════════════
with tab2:
    st.markdown("Sube un archivo Excel con múltiples registros.")

    with st.expander("📋 Ver columnas requeridas"):
        st.write("**Numéricas:**")
        for col in cols_escalar:
            st.code(col)
        st.write("**Binarias:**")
        for col in binarias_feature:
            clases = list(le_binarios[col].classes_)
            amigables = [etiqueta_amigable(c) for c in clases]
            st.code(f"{col}  →  Usar valores originales: {clases}  (equivale a: {amigables})")
        st.write("**Categóricas:**")
        for i, col in enumerate(cols_categoricas):
            cats = list(ohe.categories_[i])
            st.code(f"{col}  →  {cats}")

    archivo = st.file_uploader(
        "📂 Sube tu archivo Excel (.xlsx)",
        type=['xlsx', 'xls']
    )

    if archivo is not None:
        try:
            df_excel = pd.read_excel(archivo)
            st.success(
                f"✅ Archivo cargado: **{df_excel.shape[0]} filas** × "
                f"**{df_excel.shape[1]} columnas**"
            )

            # Vista previa con etiquetas amigables
            df_preview = df_excel.copy()
            for col in binarias_feature:
                if col in df_preview.columns:
                    df_preview[col] = df_preview[col].map(
                        lambda x: f"{x} ({etiqueta_amigable(x)})"
                    )

            st.subheader("📄 Vista previa")
            st.dataframe(df_preview.head(10), use_container_width=True)

            # Validar columnas
            columnas_requeridas = list(cols_escalar) + binarias_feature + list(cols_categoricas)
            faltantes = [c for c in columnas_requeridas if c not in df_excel.columns]

            if faltantes:
                st.error(f"❌ Columnas faltantes: {faltantes}")
                st.stop()

            extras = [c for c in df_excel.columns if c not in columnas_requeridas]
            if extras:
                st.warning(f"⚠️ Columnas extra ignoradas: {extras}")
                df_excel = df_excel[columnas_requeridas]

            st.success("✅ Validación completada")

            if st.button("🚀 Predecir todo el archivo", type="primary"):
                try:
                    df_final = preprocesar(df_excel)
                    X = df_final.values.astype(float)

                    with st.spinner(f'Procesando {len(df_excel)} registros...'):
                        predicciones = modelo.predict(X)
                        probabilidades = modelo.predict_proba(X)

                    st.success(f"✅ {len(predicciones)} predicciones completadas")

                    # Resultados con etiquetas amigables
                    clases = modelo.classes_
                    df_resultados = df_excel.copy()

                    # Reemplazar 0/1 por No/Sí en la vista
                    for col in binarias_feature:
                        if col in df_resultados.columns:
                            df_resultados[col] = df_resultados[col].map(
                                lambda x: etiqueta_amigable(x)
                            )

                    df_resultados['🎯 PREDICCIÓN'] = predicciones
                    for j, clase in enumerate(clases):
                        df_resultados[f'Prob_{clase}'] = np.round(
                            probabilidades[:, j] * 100, 2
                        )

                    st.subheader("📊 Resultados")
                    st.dataframe(df_resultados, use_container_width=True)

                    # Resumen
                    st.subheader("📈 Resumen")
                    conteo = pd.Series(predicciones).value_counts().reset_index()
                    conteo.columns = ['Clase', 'Cantidad']

                    col1, col2 = st.columns(2)
                    with col1:
                        st.dataframe(conteo, use_container_width=True)
                    with col2:
                        st.bar_chart(pd.Series(predicciones).value_counts())

                    # Descarga
                    from io import BytesIO
                    buffer = BytesIO()
                    df_resultados.to_excel(buffer, index=False)
                    buffer.seek(0)

                    st.download_button(
                        label="📥 Descargar resultados",
                        data=buffer,
                        file_name="resultados_prediccion.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    import traceback
                    st.code(traceback.format_exc())

        except Exception as e:
            st.error(f"❌ Error al leer archivo: {e}")