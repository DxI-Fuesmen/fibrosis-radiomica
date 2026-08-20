import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# 1. Cargar el dataset extraído
archivo_entrada = "dataset_radiomico_fibrosis.csv"
archivo_salida = "dataset_seleccionado_top15.csv"

print(f"Cargando {archivo_entrada}...")
df = pd.read_csv(archivo_entrada)

# 2. Separar la matriz matemática de los metadatos de control
metadatos = ['Clase_Label', 'ID_Paciente', 'Nombre_Archivo']
X = df.drop(columns=metadatos)
y = df['Clase_Label']

# Limpieza de seguridad: reemplazar posibles infinitos o nulos (común en radiómica) por 0
X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

# 3. Estandarización (Paso CRÍTICO para LASSO)
# Todas las variables deben estar en la misma escala (media 0, varianza 1)
# para que la penalización matemática sea justa.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# 4. Aplicar LASSO (Regresión Logística L1)
# El parámetro C controla la fuerza de la regularización (menor C = más variables a cero).
print("Aplicando regularización L1 (LASSO)...")
modelo_lasso = LogisticRegression(
    penalty='l1', 
    solver='liblinear', 
    random_state=42, 
    max_iter=2000,
    C=0.1 # Ajuste estándar para forzar escasez
)
modelo_lasso.fit(X_scaled_df, y)

# 5. Extraer y ordenar los coeficientes por su magnitud absoluta
coeficientes = pd.Series(modelo_lasso.coef_[0], index=X_scaled_df.columns)
coeficientes_absolutos = coeficientes.abs().sort_values(ascending=False)

# Tomar estrictamente las 15 características con mayor peso predictivo
top_15_features = coeficientes_absolutos.head(15)

print("\n--- TOP 15 CARACTERÍSTICAS SELECCIONADAS ---")
for feature, importancia in top_15_features.items():
    print(f"- {feature} (Peso: {importancia:.4f})")

# 6. Crear y guardar el nuevo dataset reducido
# Conservamos los metadatos para poder hacer GroupShuffleSplit más adelante
columnas_finales = metadatos + list(top_15_features.index)
df_reducido = df[columnas_finales]

df_reducido.to_csv(archivo_salida, index=False)
print(f"\n¡Proceso exitoso! Matriz reducida guardada en: {archivo_salida}")
print(f"Nuevas dimensiones: {df_reducido.shape[0]} parches x {df_reducido.shape[1]} columnas.")

# 7. Visualización gráfica de la importancia
plt.figure(figsize=(12, 8))
# Invertir el orden para que la más importante quede arriba en el gráfico
top_15_features.sort_values(ascending=True).plot(kind='barh', color='teal')
plt.title('Top 15 Características Radiómicas (Regularización LASSO)')
plt.xlabel('Magnitud Absoluta del Coeficiente')
plt.ylabel('Característica Radiómica')
plt.tight_layout()
plt.show()