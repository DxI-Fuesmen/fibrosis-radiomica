import os
import re
import random
import pandas as pd
import numpy as np
import SimpleITK as sitk
from radiomics import featureextractor
import warnings

# Ignorar la advertencia inofensiva de ITK en la consola para no ensuciar la vista
warnings.filterwarnings("ignore")

# ==========================================
# 1. CONFIGURACIÓN DE RUTAS Y PARÁMETROS
# ==========================================
# Reemplaza estas rutas por la ubicación real de tus carpetas en Windows
# Importante: Usa doble barra \\ en las rutas de Windows o pon una 'r' antes de las comillas
dir_healthy = r"C:\Users\rtagliavini\Documents\Base Pulmon\ILD_DB\Talisman_Clasificado\healthy"
dir_fibrosis = r"C:\Users\rtagliavini\Documents\Base Pulmon\ILD_DB\Talisman_Clasificado\fibrosis"

# Archivo CSV de salida con todas las características
output_csv = "dataset_radiomico_fibrosis.csv"

# Límite de parches para balancear las clases (tienes 2793 de fibrosis, así que 2500 es un buen número)
MAX_PATCHES_POR_CLASE = 2500 

def extraer_id_paciente(nombre_archivo):
    """Extrae el ID del paciente (ej: patient64) para evitar Data Leakage después"""
    match = re.search(r'patient(-?\d+)', nombre_archivo)
    return match.group(1) if match else "desconocido"

def recopilar_archivos(directorio, etiqueta, limite):
    archivos = [f for f in os.listdir(directorio) if f.endswith('.tif')]
    random.shuffle(archivos)
    
    if limite and len(archivos) > limite:
        archivos = archivos[:limite]
        
    lista_datos = []
    for f in archivos:
        lista_datos.append({
            'path': os.path.join(directorio, f),
            'class': etiqueta,      # 0 = sano, 1 = fibrosis
            'patient_id': extraer_id_paciente(f)
        })
    return lista_datos

def iniciar_extractor():
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.enableAllImageTypes() # Wavelet, LoG, etc.
    extractor.enableAllFeatures()   # Firstorder, GLCM, GLRLM, etc.
    extractor.settings['binWidth'] = 25.0
    extractor.settings['minimumROIDimensions'] = 2
    return extractor

def main():
    print("Recolectando y balanceando los parches...")
    sanos = recopilar_archivos(dir_healthy, etiqueta=0, limite=MAX_PATCHES_POR_CLASE)
    fibrosis = recopilar_archivos(dir_fibrosis, etiqueta=1, limite=MAX_PATCHES_POR_CLASE)
    
    dataset_total = sanos + fibrosis
    random.shuffle(dataset_total)
    
    print(f"Total a procesar: {len(dataset_total)} (Sanos: {len(sanos)}, Fibrosis: {len(fibrosis)})")
    extractor = iniciar_extractor()
    resultados = []
    
    print("Iniciando extracción con PyRadiomics. Esto puede tomar bastante tiempo...")
    for i, item in enumerate(dataset_total):
        try:
            imagen_sitk = sitk.ReadImage(item['path'])
            # Crear máscara binaria que cubra todo el parche (puros 1s)
            matriz_imagen = sitk.GetArrayFromImage(imagen_sitk)
            mascara_sitk = sitk.GetImageFromArray(np.ones_like(matriz_imagen, dtype=np.uint8))
            mascara_sitk.CopyInformation(imagen_sitk)
            
            # Extraer características radiómicas
            features = extractor.execute(imagen_sitk, mascara_sitk)
            
            # Limpiar datos y conservar solo los valores numéricos
            clean_features = {k: v for k, v in features.items() if not k.startswith('diagnostics_')}
            clean_features['Clase_Label'] = item['class']
            clean_features['ID_Paciente'] = item['patient_id']
            clean_features['Nombre_Archivo'] = os.path.basename(item['path'])
            
            resultados.append(clean_features)
            
            if (i + 1) % 100 == 0:
                print(f"-> Procesados {i + 1} de {len(dataset_total)} parches...")
                
        except Exception as e:
            print(f"Error en {item['path']}: {e}")
            
    df = pd.DataFrame(resultados)
    df.to_csv(output_csv, index=False)
    print(f"\n¡Listo! Matriz guardada en: {output_csv}")
    print(f"Dimensiones finales: {df.shape[0]} parches x {df.shape[1]} variables.")

if __name__ == "__main__":
    main()