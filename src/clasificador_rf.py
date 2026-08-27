import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, 
                             roc_auc_score, RocCurveDisplay, ConfusionMatrixDisplay)
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 1. CONFIGURACIÓN DE RUTAS
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

archivo_entrada = os.path.join(DATA_DIR, "dataset_seleccionado_top15.csv")
archivo_modelo = os.path.join(MODELS_DIR, "modelo_rf_fibrosis.pkl")
archivo_grafico = os.path.join(DATA_DIR, "evaluacion_rf.png")

print(f"Cargando {archivo_entrada}...", flush=True)
df = pd.read_csv(archivo_entrada)

# ==========================================
# 2. SEPARAR VARIABLES
# ==========================================
metadatos = ['Clase_Label', 'ID_Paciente', 'Nombre_Archivo']
X = df.drop(columns=metadatos)
y = df['Clase_Label']
grupos = df['ID_Paciente']
features_list = list(X.columns)

# ==========================================
# 3. PARTICIÓN ESTRATIFICADA POR PACIENTE
# ==========================================
print("Realizando particion de datos estratificada por paciente y clase...", flush=True)
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

for train_idx, test_idx in sgkf.split(X, y, groups=grupos):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    grupos_train, grupos_test = grupos.iloc[train_idx], grupos.iloc[test_idx]
    break 

print(f"Parches de Entrenamiento: {len(X_train)} (Pacientes: {grupos_train.nunique()})", flush=True)
print(f"Parches de Prueba:        {len(X_test)} (Pacientes: {grupos_test.nunique()})", flush=True)

# ==========================================
# 4. ENTRENAR MODELO RANDOM FOREST
# ==========================================
print("\nEntrenando el modelo Random Forest...", flush=True)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
rf_model.fit(X_train, y_train)

# ==========================================
# 5. EVALUACIÓN DE DESEMPEÑO
# ==========================================
y_pred = rf_model.predict(X_test)
y_proba = rf_model.predict_proba(X_test)[:, 1] 

exactitud = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print("\n" + "="*40, flush=True)
print("RESULTADOS DEL MODELO RANDOM FOREST", flush=True)
print("="*40, flush=True)
print(f"Accuracy (Exactitud): {exactitud:.4f}", flush=True)
print(f"ROC-AUC Score:        {auc:.4f}\n", flush=True)
print("Reporte de Clasificacion Detallado:", flush=True)
print(classification_report(y_test, y_pred, target_names=['Sano (0)', 'Fibrosis (1)']), flush=True)

# ==========================================
# 6. VISUALIZACIÓN DE RESULTADOS
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=['Sano', 'Fibrosis'], cmap='Blues', ax=ax1)
ax1.set_title('Matriz de Confusion (Conjunto Test)')

RocCurveDisplay.from_estimator(rf_model, X_test, y_test, ax=ax2, color='darkorange')
ax2.plot([0, 1], [0, 1], color='navy', linestyle='--') 
ax2.set_title(f'Curva ROC (AUC = {auc:.3f})')

plt.tight_layout()
plt.savefig(archivo_grafico, dpi=300, bbox_inches='tight')
print(f"- Grafico de evaluacion guardado en: {archivo_grafico}", flush=True)
plt.close()

# ==========================================
# 7. GUARDAR MODELO SERIALIZADO
# ==========================================
model_artifact = {
    'model': rf_model,
    'features': features_list,
    'accuracy': exactitud,
    'roc_auc': auc
}
joblib.dump(model_artifact, archivo_modelo)
print(f"\nModelo guardado exitosamente en: {archivo_modelo}!", flush=True)