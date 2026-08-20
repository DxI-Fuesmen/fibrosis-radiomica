import os
import joblib
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
from radiomics import featureextractor
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 1. CONFIGURACIÓN
# ==========================================
# Ruta a UN archivo DICOM (.dcm) de prueba
ruta_dicom = r"C:\Users\rtagliavini\Documents\Base Pulmon\ILD_DB\ILD_DB_Clasificada\healthy\47\CT-9465-0020.dcm"

# Cargar el modelo que guardaste en el paso anterior
modelo_path = "modelo_rf_fibrosis.pkl"
rf_model = joblib.load(modelo_path)

# IMPORTANTE: Pegá acá exactamente las 15 características que seleccionó LASSO, 
# en el MISMO ORDEN en el que aparecen en tu dataset_seleccionado_top15.csv
top_15_features = [
    'wavelet-LL_ngtdm_Coarseness',
    'wavelet-HL_firstorder_Median',
    'wavelet-LL_glcm_InverseVariance',
    'wavelet-LL_glrlm_GrayLevelNonUniformity',
    'square_glcm_ClusterShade',
    'wavelet-LL_gldm_SmallDependenceEmphasis',
    'original_gldm_LowGrayLevelEmphasis',
    'square_glcm_SumEntropy',
    'wavelet-LL_firstorder_Uniformity',
    'wavelet-LH_firstorder_Median',
    'square_glszm_ZoneEntropy',
    'wavelet-LL_glcm_SumEntropy',
    'gradient_glcm_Imc1',
    'wavelet-LL_glcm_ClusterShade',
    'wavelet-LL_glrlm_GrayLevelNonUniformityNormalized'
]

# Parámetros de la ventana
WINDOW_SIZE = 32
# STRIDE (Salto): 32 = rápido pero en bloques. 16 o 8 = más lento pero mejor resolución.
STRIDE = 16 

# ==========================================
# 2. INICIAR EXTRACTOR
# ==========================================
extractor = featureextractor.RadiomicsFeatureExtractor()
extractor.enableAllImageTypes()
extractor.enableAllFeatures()
extractor.settings['binWidth'] = 25.0
extractor.settings['minimumROIDimensions'] = 2

# ==========================================
# 3. PROCESAMIENTO DE LA IMAGEN
# ==========================================
print("Cargando imagen DICOM...")
imagen_sitk = sitk.ReadImage(ruta_dicom)
matriz_img = sitk.GetArrayFromImage(imagen_sitk)[0] # Tomar el corte 2D
alto, ancho = matriz_img.shape

# Matriz vacía para guardar las probabilidades
heatmap = np.zeros((alto, ancho))
# Matriz para contar cuántas ventanas pasaron por cada píxel (para promediar)
conteo_ventanas = np.zeros((alto, ancho))

print(f"Iniciando escaneo (Resolución: {ancho}x{alto}, Ventana: {WINDOW_SIZE}, Salto: {STRIDE})")
print("Esto puede tomar varios minutos por la extracción matemática en vivo...")

# Recorrer la imagen
for y in range(0, alto - WINDOW_SIZE + 1, STRIDE):
    for x in range(0, ancho - WINDOW_SIZE + 1, STRIDE):
        
        # 1. Recortar el parche
        parche_array = matriz_img[y:y+WINDOW_SIZE, x:x+WINDOW_SIZE]
        
        # Omitir fondo puro (si el parche es todo aire externo, evitamos calcular)
        if np.mean(parche_array) < -950:
            continue
            
        parche_sitk = sitk.GetImageFromArray(parche_array)
        mascara_sitk = sitk.GetImageFromArray(np.ones_like(parche_array, dtype=np.uint8))
        
        # 2. Extraer características
        try:
            features = extractor.execute(parche_sitk, mascara_sitk)
            
            # 3. Filtrar solo las Top 15 y ordenarlas para el modelo
            datos_parche = []
            for feat in top_15_features:
                # Si la característica no se pudo extraer (ej. por bordes raros), asignar 0
                valor = features.get(feat, 0.0) 
                datos_parche.append(valor)
                
            X_nuevo = pd.DataFrame([datos_parche], columns=top_15_features)
            
            # 4. Predecir probabilidad de Fibrosis (Clase 1)
            proba_fibrosis = rf_model.predict_proba(X_nuevo)[0][1]
            
            # 5. Sumar probabilidad al mapa y registrar la superposición
            heatmap[y:y+WINDOW_SIZE, x:x+WINDOW_SIZE] += proba_fibrosis
            conteo_ventanas[y:y+WINDOW_SIZE, x:x+WINDOW_SIZE] += 1
            
        except Exception:
            continue
            
    print(f"Fila {y}/{alto - WINDOW_SIZE} procesada...")

# Promediar las superposiciones
conteo_ventanas[conteo_ventanas == 0] = 1 # Evitar división por cero
heatmap_final = heatmap / conteo_ventanas

# ==========================================
# 4. VISUALIZACIÓN CLÍNICA
# ==========================================
plt.figure(figsize=(10, 10))
# Mostrar imagen original de fondo (blanco y negro)
plt.imshow(matriz_img, cmap='gray', vmin=-1000, vmax=400)
# Superponer el mapa de calor (usamos 'jet': azul=sano, rojo=fibrosis)
plt.imshow(heatmap_final, cmap='jet', alpha=0.4)
plt.colorbar(label='Probabilidad Predictiva de Fibrosis')
plt.title(f'Mapa Radiómico de Fibrosis - Salto: {STRIDE}px')
plt.axis('off')
plt.show()