import os
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURACIÓN DE RUTAS
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

archivo_entrada = os.path.join(DATA_DIR, "dataset_radiomico_fibrosis.csv")
archivo_salida = os.path.join(DATA_DIR, "dataset_seleccionado_top15.csv")
archivo_json_features = os.path.join(DATA_DIR, "selected_features.json")
archivo_grafico = os.path.join(DATA_DIR, "lasso_top15_features.png")

print(f"Cargando {archivo_entrada}...", flush=True)
df = pd.read_csv(archivo_entrada)

# ==========================================
# 2. SEPARAR MATRIZ Y METADATOS
# ==========================================
metadatos = ['Clase_Label', 'ID_Paciente', 'Nombre_Archivo']
X = df.drop(columns=metadatos)
y = df['Clase_Label']

# Limpieza de seguridad: reemplazar posibles infinitos o nulos por 0
X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

# ==========================================
# 3. ESTANDARIZACIÓN
# ==========================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# ==========================================
# 4. REGULARIZACIÓN L1 (LASSO)
# ==========================================
print("Aplicando regularización L1 (LASSO)...", flush=True)
modelo_lasso = LogisticRegression(
    penalty='l1', 
    solver='liblinear', 
    random_state=42, 
    max_iter=2000,
    C=0.1 # Menor C = mayor escasez (menos variables distintas de cero)
)
modelo_lasso.fit(X_scaled_df, y)

# ==========================================
# 5. EXTRAER TOP 15 CARACTERÍSTICAS
# ==========================================
coeficientes = pd.Series(modelo_lasso.coef_[0], index=X_scaled_df.columns)
coeficientes_absolutos = coeficientes.abs().sort_values(ascending=False)

top_15_features = coeficientes_absolutos.head(15)

print("\n--- TOP 15 CARACTERÍSTICAS SELECCIONADAS ---", flush=True)
feature_details = {}
image_types_needed = set()
feature_classes_needed = set()

for feature, importancia in top_15_features.items():
    coef_val = float(coeficientes[feature])
    print(f"- {feature} (Coeficiente: {coef_val:+.4f}, Magnitud: {importancia:.4f})", flush=True)
    
    # Descomponer nombre para deducir filtros requeridos en PyRadiomics
    # Formato típico: {imageType}_{featureClass}_{featureName} o wavelet-{band}_{featureClass}_{featureName}
    parts = feature.split('_')
    img_type = parts[0]
    feat_class = parts[1] if len(parts) > 1 else "firstorder"
    
    if img_type.startswith("wavelet"):
        image_types_needed.add("Wavelet")
    elif img_type == "original":
        image_types_needed.add("Original")
    elif img_type == "gradient":
        image_types_needed.add("Gradient")
    elif img_type == "square":
        image_types_needed.add("Square")
    elif img_type == "logarithm":
        image_types_needed.add("Logarithm")
    elif img_type == "exponential":
        image_types_needed.add("Exponential")
    elif img_type == "squareroot":
        image_types_needed.add("Squareroot")
    elif img_type == "log":
        image_types_needed.add("LoG")
    else:
        image_types_needed.add(img_type.capitalize())
        
    feature_classes_needed.add(feat_class)
    
    feature_details[feature] = {
        "coefficient": coef_val,
        "importance": float(importancia),
        "image_type": img_type,
        "feature_class": feat_class
    }

# ==========================================
# 6. GUARDAR DATASET REDUCIDO Y METADATA JSON
# ==========================================
columnas_finales = metadatos + list(top_15_features.index)
df_reducido = df[columnas_finales]
df_reducido.to_csv(archivo_salida, index=False)

metadata_payload = {
    "n_features": len(top_15_features),
    "features": list(top_15_features.index),
    "feature_details": feature_details,
    "required_image_types": sorted(list(image_types_needed)),
    "required_feature_classes": sorted(list(feature_classes_needed)),
    "metadata_columns": metadatos
}

with open(archivo_json_features, 'w', encoding='utf-8') as f:
    json.dump(metadata_payload, f, indent=4, ensure_ascii=False)

print(f"\n¡Proceso exitoso!", flush=True)
print(f"- Matriz reducida guardada en: {archivo_salida}", flush=True)
print(f"- Metadatos JSON guardados en: {archivo_json_features}", flush=True)
print(f"Dimensiones: {df_reducido.shape[0]} parches x {df_reducido.shape[1]} columnas.", flush=True)

# ==========================================
# 7. VISUALIZACIÓN GRÁFICA
# ==========================================
plt.figure(figsize=(12, 8))
top_15_features.sort_values(ascending=True).plot(kind='barh', color='teal')
plt.title('Top 15 Características Radiómicas (Regularización LASSO)')
plt.xlabel('Magnitud Absoluta del Coeficiente')
plt.ylabel('Característica Radiómica')
plt.tight_layout()
plt.savefig(archivo_grafico, dpi=300, bbox_inches='tight')
print(f"- Gráfico de importancia guardado en: {archivo_grafico}", flush=True)
plt.close()