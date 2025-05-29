import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

# ----- 1) Cargar datos y modelos -----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

autos_limpios = pd.read_csv(os.path.join(DATA_DIR, 'autos_limpios.csv'))

with open(os.path.join(DATA_DIR, 'mejor_modelo_xgb.pkl'), 'rb') as f:
    modelo = pickle.load(f)

with open(os.path.join(DATA_DIR, 'columnas_modelo.pkl'), 'rb') as f:
    columnas_modelo = pickle.load(f)

with open(os.path.join(DATA_DIR, 'scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)

# ----- 2) Sidebar Inputs -----
st.sidebar.header("Ingresá las características del auto")

marca = st.sidebar.selectbox("Marca", sorted(autos_limpios['marca'].str.title().unique()))
marca_lower = marca.lower()

modelos_disp = autos_limpios[autos_limpios['marca'] == marca_lower]['modelo'].unique()
modelo = st.sidebar.selectbox("Modelo", sorted(modelos_disp)).lower()

motores = autos_limpios[autos_limpios['modelo'] == modelo]['motor'].unique()
motor = st.sidebar.selectbox("Motor (L)", sorted(motores))

transmisiones = autos_limpios[autos_limpios['modelo'] == modelo]['transmision'].dropna().unique()
transmision = st.sidebar.selectbox("Transmisión", sorted(transmisiones)).lower()

tracciones = autos_limpios[autos_limpios['modelo'] == modelo]['traccion'].dropna().unique()
traccion = st.sidebar.selectbox("Tracción", sorted(tracciones)).lower()

ano = st.sidebar.number_input("Año", min_value=1980, max_value=2025, value=2020)
kilometros = st.sidebar.number_input("Kilómetros", min_value=0, max_value=500000, value=80000, step=1000)

if st.sidebar.button("Predecir Precio"):

    # ----- 3) Preparar datos -----
    X_pred = pd.DataFrame(0, index=[0], columns=columnas_modelo)
    X_pred.loc[0, 'ano'] = ano
    X_pred.loc[0, 'motor'] = float(motor)
    for var, val in {
        'marca': marca_lower,
        'modelo': modelo,
        'transmision': transmision,
        'traccion': traccion
    }.items():
        col = f"{var}_{val}"
        if col in X_pred.columns:
            X_pred.loc[0, col] = 1

    # ----- 4) Curva de precios -----
    km_min = max(1000, kilometros - 100000)
    km_max = kilometros + 100000
    num_puntos = 30
    curva_kms = np.linspace(km_min, km_max, num=num_puntos, dtype=int).tolist()
    curva_precios = []
    for km in curva_kms:
        X_tmp = X_pred.copy()
        X_tmp.loc[0, 'kilometros'] = km
        X_tmp[['ano', 'kilometros', 'motor']] = scaler.transform(X_tmp[['ano', 'kilometros', 'motor']])
        curva_precios.append(float(modelo.predict(X_tmp)[0]))

    # Predicción base y +10k
    X_tmp = X_pred.copy()
    X_tmp.loc[0, 'kilometros'] = kilometros
    X_tmp[['ano', 'kilometros', 'motor']] = scaler.transform(X_tmp[['ano', 'kilometros', 'motor']])
    precio_base = float(modelo.predict(X_tmp)[0])
    paso = (km_max - km_min) / (num_puntos - 1)
    idx_10k = min(len(curva_kms) - 1, int((kilometros + 10000 - km_min) / paso))
    delta_10000km = curva_precios[idx_10k] - precio_base

    # ----- 5) Mostrar resultados -----
    st.markdown(f"### Precio estimado: **ARS {precio_base:,.0f}**")
    st.markdown(f"**Cambio estimado por +10 000 km:** ARS {delta_10000km:,.0f}")

    # ----- 6) Gráfico -----
    fig, ax = plt.subplots()
    ax.plot(curva_kms, curva_precios, marker='o')
    ax.set_title("Curva de precio vs kilómetros")
    ax.set_xlabel("Kilómetros")
    ax.set_ylabel("Precio estimado (ARS)")
    st.pyplot(fig)

    # ----- 7) Importancias del modelo -----
    booster_score = modelo.get_booster().get_score(importance_type='weight')
    import_dict = {col: 0 for col in columnas_modelo}
    for feat, imp in booster_score.items():
        if feat in import_dict:
            import_dict[feat] = imp
    df_imp = pd.DataFrame({
        'feature': list(import_dict.keys()),
        'importance': list(import_dict.values())
    })

    sel_cols = [f"marca_{marca_lower}", f"modelo_{modelo}"]
    df_fil = df_imp[
        (~df_imp['feature'].str.startswith('marca_') & 
         ~df_imp['feature'].str.startswith('modelo_')) |
        df_imp['feature'].isin(sel_cols)
    ]
    total = df_fil['importance'].sum() or 1
    df_fil['importance_pct'] = (df_fil['importance']/total*100).round(2)
    df_fil = df_fil.sort_values('importance_pct', ascending=False)

    st.markdown("### Importancia de las variables")
    st.dataframe(df_fil[['feature', 'importance_pct']].rename(columns={
        'feature': 'Variable',
        'importance_pct': '% Importancia'
    }))
