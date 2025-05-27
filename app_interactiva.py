import os
from flask import Flask, render_template, request, jsonify, send_from_directory
import pandas as pd
import pickle
import numpy as np

# ----- 1) Definir rutas base -----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
TEMPLATES_DIR = os.path.join(BASE_DIR, 'templates')
STATIC_DIR = os.path.join(BASE_DIR, 'static')

# ----- 2) Cargar datos, modelo y columnas desde rutas relativas -----
autos_limpios = pd.read_csv(os.path.join(DATA_DIR, 'autos_limpios.csv'))

with open(os.path.join(DATA_DIR, 'mejor_modelo_xgb.pkl'), 'rb') as f:
    modelo = pickle.load(f)

with open(os.path.join(DATA_DIR, 'columnas_modelo.pkl'), 'rb') as f:
    columnas_modelo = pickle.load(f)

with open(os.path.join(DATA_DIR, 'scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)

# ----- 3) Inicializar Flask apuntando a las carpetas relativas -----
app = Flask(
    __name__,
    template_folder=TEMPLATES_DIR,
    static_folder=STATIC_DIR
)

# Página inicial
@app.route('/')
def home():
    marcas = sorted(autos_limpios['marca'].unique())
    marcas_capitalizadas = [m.title() for m in marcas]
    return render_template('index_interactiva.html', marcas=marcas_capitalizadas)

# Endpoints AJAX
@app.route('/get_modelos', methods=['POST'])
def get_modelos():
    marca = request.form['marca'].lower()
    modelos = sorted(autos_limpios[autos_limpios['marca'] == marca]['modelo'].unique())
    return jsonify(modelos)

@app.route('/get_motores', methods=['POST'])
def get_motores():
    modelo_nombre = request.form['modelo']
    motores = sorted(autos_limpios[autos_limpios['modelo'] == modelo_nombre]['motor'].unique())
    return jsonify([str(m) for m in motores])

@app.route('/get_transmisiones', methods=['POST'])
def get_transmisiones():
    modelo_nombre = request.form['modelo']
    transmisiones = sorted(
        autos_limpios[autos_limpios['modelo'] == modelo_nombre]['transmision']
        .dropna()
        .unique()
    )
    return jsonify(transmisiones)

@app.route('/get_tracciones', methods=['POST'])
def get_tracciones():
    modelo_nombre = request.form['modelo']
    tracciones = sorted(
        autos_limpios[autos_limpios['modelo'] == modelo_nombre]['traccion']
        .dropna()
        .unique()
    )
    return jsonify(tracciones)

# Ruta para predicción
@app.route('/predict', methods=['POST'])
def predict():
    # 1) Leer inputs
    marca = request.form['marca'].lower()
    modelo_nombre = request.form['modelo'].lower()
    transmision = request.form['transmision'].lower()
    traccion = request.form['traccion'].lower()
    ano = int(request.form['ano'])
    km_usuario = int(request.form['kilometros'])
    motor = float(request.form['motor'])

    # 2) Crear DataFrame base y asignar valores
    X_pred = pd.DataFrame(0, index=[0], columns=columnas_modelo)
    X_pred.loc[0, 'ano'] = ano
    X_pred.loc[0, 'motor'] = motor
    for var, val in {
        'marca': marca,
        'modelo': modelo_nombre,
        'transmision': transmision,
        'traccion': traccion
    }.items():
        col = f"{var}_{val}"
        if col in X_pred.columns:
            X_pred.loc[0, col] = 1

    # 3) Generar la curva de precios
    km_min = max(1000, km_usuario - 100000)
    km_max = km_usuario + 100000
    num_puntos = 30
    curva_kms = np.linspace(km_min, km_max, num=num_puntos, dtype=int).tolist()
    curva_precios = []
    for km in curva_kms:
        X_tmp = X_pred.copy()
        X_tmp.loc[0, 'kilometros'] = km
        X_tmp[['ano','kilometros','motor']] = scaler.transform(
            X_tmp[['ano','kilometros','motor']])
        curva_precios.append(float(modelo.predict(X_tmp)[0]))

    # 4) Predicción puntual y delta +10k
    X_tmp = X_pred.copy()
    X_tmp.loc[0, 'kilometros'] = km_usuario
    X_tmp[['ano','kilometros','motor']] = scaler.transform(
        X_tmp[['ano','kilometros','motor']])
    precio_base = float(modelo.predict(X_tmp)[0])

    paso = (km_max - km_min) / (num_puntos - 1)
    idx_10k = min(len(curva_kms)-1,
                   int((km_usuario + 10000 - km_min) / paso))
    delta_10000km = curva_precios[idx_10k] - precio_base

    resultado = (
        f"Predicción para {marca.title()} - {modelo_nombre.title()} "
        f"con {km_usuario:,} km del {ano}: $ARS {precio_base:,.0f}"
        "<br><br>"
        f"↓ Cambio por +10 000 km: $ARS {delta_10000km:,.0f}"
    ).replace(',', ' ')

    # 5) Calcular importancias completas, filtrar y normalizar
    booster_score = modelo.get_booster().get_score(importance_type='weight')
    import_dict = {col: 0 for col in columnas_modelo}
    for feat, imp in booster_score.items():
        if feat in import_dict:
            import_dict[feat] = imp
    df_imp = pd.DataFrame({
        'feature': list(import_dict.keys()),
        'importance': list(import_dict.values())
    })
    sel_cols = [f"marca_{marca}", f"modelo_{modelo_nombre}"]
    df_fil = df_imp[
        (~df_imp['feature'].str.startswith('marca_') & 
         ~df_imp['feature'].str.startswith('modelo_'))
        | df_imp['feature'].isin(sel_cols)
    ]
    total = df_fil['importance'].sum() or 1
    df_fil['importance_pct'] = (df_fil['importance']/total*100).round(2)
    df_fil = df_fil.sort_values('importance_pct', ascending=False)

    features = df_fil['feature'].tolist()
    importancias = df_fil['importance_pct'].tolist()

    # 7) Renderizar resultado
    return render_template(
        'result.html',
        resultado=resultado,
        curva_kms=curva_kms,
        curva_precios=curva_precios,
        kilometros=km_usuario,
        precio=precio_base,
        delta_10000km=delta_10000km,
        features=features,
        importancias=importancias,
        marca=marca,
        modelo=modelo_nombre,
        transmision=transmision,
        traccion=traccion
    )

# Ruta para ver importancias generales
@app.route('/importancias')
def mostrar_importancias():
    booster_score = modelo.get_booster().get_score(importance_type='weight')
    import_dict = {col: 0 for col in columnas_modelo}
    for feat, imp in booster_score.items():
        if feat in import_dict:
            import_dict[feat] = imp
    df = pd.DataFrame({
        'feature': list(import_dict.keys()),
        'importance': list(import_dict.values())
    }).sort_values('importance', ascending=False)
    return render_template(
        'importancias.html',
        tabla=df.to_html(index=False)
    )

if __name__ == '__main__':
    app.run(debug=True)





