import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, 
                             roc_auc_score, RocCurveDisplay, ConfusionMatrixDisplay)
import warnings

warnings.filterwarnings("ignore")

# 1. Cargar el dataset reducido
archivo = "dataset_seleccionado_top15.csv"
print(f"Cargando {archivo}...")
df = pd.read_csv(archivo)

# 2. Separar variables
metadatos = ['Clase_Label', 'ID_Paciente', 'Nombre_Archivo']
X = df.drop(columns=metadatos)
y = df['Clase_Label']
grupos = df['ID_Paciente']

# 3. Partición de Datos Estratificada por Paciente y Clase
print("Realizando partición de datos estratificada por paciente y clase...")
# Dividimos en 5 grupos (folds), lo que deja 80% train y 20% test, asegurando balance
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

# Tomamos solo la primera partición generada
for train_idx, test_idx in sgkf.split(X, y, groups=grupos):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    grupos_train, grupos_test = grupos.iloc[train_idx], grupos.iloc[test_idx]
    break 

print(f"Parches de Entrenamiento: {len(X_train)} (Pacientes: {grupos_train.nunique()})")
print(f"Parches de Prueba: {len(X_test)} (Pacientes: {grupos_test.nunique()})")

# 4. Configurar y Entrenar el Modelo Random Forest
print("\nEntrenando el modelo Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
rf_model.fit(X_train, y_train)

# 5. Predicciones y Evaluación
y_pred = rf_model.predict(X_test)
y_proba = rf_model.predict_proba(X_test)[:, 1] 

exactitud = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print("\n" + "="*40)
print("🎯 RESULTADOS DEL MODELO")
print("="*40)
print(f"Accuracy (Exactitud): {exactitud:.4f}")
print(f"ROC-AUC Score:        {auc:.4f}\n")
print("Reporte de Clasificación Detallado:")
print(classification_report(y_test, y_pred, target_names=['Sano (0)', 'Fibrosis (1)']))

# 6. Visualización de Resultados
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=['Sano', 'Fibrosis'], cmap='Blues', ax=ax1)
ax1.set_title('Matriz de Confusión')

RocCurveDisplay.from_estimator(rf_model, X_test, y_test, ax=ax2, color='darkorange')
ax2.plot([0, 1], [0, 1], color='navy', linestyle='--') 
ax2.set_title('Curva ROC (Receiver Operating Characteristic)')

plt.tight_layout()
plt.show()

import joblib
# Guardar el modelo entrenado para usarlo en nuevos pacientes
joblib.dump(rf_model, 'modelo_rf_fibrosis.pkl')
print("Modelo guardado como 'modelo_rf_fibrosis.pkl'")