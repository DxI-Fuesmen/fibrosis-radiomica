import os
import re
import glob
import json
import time
import joblib
import logging
import warnings
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from radiomics import featureextractor
import radiomics

warnings.filterwarnings("ignore")
radiomics.setVerbosity(logging.ERROR)
logging.getLogger("radiomics").setLevel(logging.ERROR)

# ==========================================
# 1. CONFIGURACIÓN Y RUTAS
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")

# Ruta a UN archivo DICOM de prueba (Paciente 77, corte 12 o 17)
RUTA_DICOM_DEFAULT = r"C:\Users\malcalde\Documents\Base Pulmon\ILD_DB\ILD_DB_Clasificada\fibrosis\77\CT-0002-0012.dcm"

# Archivos del modelo y características
ARCHIVO_MODELO = os.path.join(MODELS_DIR, "modelo_rf_fibrosis.pkl")
ARCHIVO_JSON_FEAT = os.path.join(DATA_DIR, "selected_features.json")

# Parámetros de la ventana
WINDOW_SIZE = 32
STRIDE = 8  # Salto en píxeles (8 px con suavizado gaussiano produce mapas de alta definición)
MIN_LUNG_PERCENT = 0.40  # Umbral mínimo de parénquima pulmonar para evitar artefactos de borde


def cargar_modelo_y_features():
    """Carga dinámicamente el modelo y las características seleccionadas por LASSO."""
    if not os.path.exists(ARCHIVO_MODELO):
        raise FileNotFoundError(f"No se encontró el modelo en: {ARCHIVO_MODELO}. Ejecuta clasificador_rf.py primero.")
    if not os.path.exists(ARCHIVO_JSON_FEAT):
        raise FileNotFoundError(f"No se encontró el archivo JSON en: {ARCHIVO_JSON_FEAT}. Ejecuta seleccion_lasso.py primero.")
        
    artifact = joblib.load(ARCHIVO_MODELO)
    rf_model = artifact['model'] if isinstance(artifact, dict) and 'model' in artifact else artifact
    
    with open(ARCHIVO_JSON_FEAT, 'r', encoding='utf-8') as f:
        meta_features = json.load(f)
        
    return rf_model, meta_features


def extraer_caracteristicas_parche(parche_array, meta_features):
    """Extrae únicamente las características seleccionadas para un parche 2D."""
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.disableAllImageTypes()
    extractor.disableAllFeatures()
    
    for img_type in meta_features.get('required_image_types', []):
        try:
            extractor.enableImageTypeByName(img_type)
        except Exception:
            extractor.enableImageTypeByName(img_type.capitalize())
            
    features_by_class = {}
    for feat_name in meta_features.get('features', []):
        parts = feat_name.split('_')
        feat_cls = parts[1]
        raw_name = parts[2]
        features_by_class.setdefault(feat_cls, []).append(raw_name)
        
    for feat_cls, names in features_by_class.items():
        extractor.enableFeaturesByName(**{feat_cls: list(set(names))})
        
    extractor.settings['binWidth'] = 25.0
    extractor.settings['minimumROIDimensions'] = 2
    
    sitk_img = sitk.GetImageFromArray(parche_array)
    sitk_mask = sitk.GetImageFromArray(np.ones_like(parche_array, dtype=np.uint8))
    
    try:
        res = extractor.execute(sitk_img, sitk_mask)
        return [float(res.get(k, 0.0)) for k in meta_features['features']]
    except Exception:
        return [0.0] * len(meta_features['features'])


def buscar_mascaras_asociadas(ruta_dicom):
    """Busca automáticamente la máscara de pulmón y la máscara ROI del corte."""
    nombre_archivo = os.path.basename(ruta_dicom)
    dir_paciente = os.path.dirname(ruta_dicom)
    
    match_corte = re.search(r'(\d+)\.dcm$', nombre_archivo, re.IGNORECASE)
    idx_corte = int(match_corte.group(1)) if match_corte else 1
    
    partes = ruta_dicom.replace('\\', '/').split('/')
    paciente_id = partes[-2] if len(partes) >= 2 else "77"
    
    base_ild = r"C:\Users\malcalde\Documents\Base Pulmon\ILD_DB"
    dir_lung = os.path.join(base_ild, "ILD_DB_lungMasks", paciente_id, "lung_mask")
    patron_lung = os.path.join(dir_lung, f"lung_mask_*_{idx_corte}.dcm")
    match_lung = glob.glob(patron_lung)
    ruta_lung = match_lung[0] if match_lung else None
    
    dir_roi = os.path.join(dir_paciente, "roi_mask")
    patron_roi = os.path.join(dir_roi, f"roi_mask_*_{idx_corte}.dcm")
    match_roi = glob.glob(patron_roi)
    ruta_roi = match_roi[0] if match_roi else None
    
    return ruta_lung, ruta_roi, idx_corte, paciente_id


def generar_mapa_2d(ruta_dicom=RUTA_DICOM_DEFAULT, stride=STRIDE, window_size=WINDOW_SIZE, min_lung_percent=MIN_LUNG_PERCENT):
    t_inicio = time.time()
    print("="*60, flush=True)
    print(f"ANÁLISIS RADIÓMICO 2D - CORTE ÚNICO (CORREGIDO)", flush=True)
    print(f"Corte: {os.path.basename(ruta_dicom)}", flush=True)
    print(f"Salto (Stride): {stride} px | Ventana: {window_size}x{window_size} px", flush=True)
    print("="*60, flush=True)
    
    rf_model, meta_features = cargar_modelo_y_features()
    top_features = meta_features['features']
    
    ruta_lung, ruta_roi, idx_corte, paciente_id = buscar_mascaras_asociadas(ruta_dicom)
    
    # 1. Cargar imagen CT
    print("Cargando imagen DICOM...", flush=True)
    img_sitk = sitk.ReadImage(ruta_dicom)
    matriz_img = sitk.GetArrayFromImage(img_sitk)[0]
    alto, ancho = matriz_img.shape
    
    # 2. Cargar máscara pulmonar
    if ruta_lung and os.path.exists(ruta_lung):
        print(f"Máscara pulmonar: {os.path.basename(ruta_lung)}", flush=True)
        lung_sitk = sitk.ReadImage(ruta_lung)
        arr_lung = sitk.GetArrayFromImage(lung_sitk)[0]
        mask_lung_bin = (arr_lung > 0).astype(np.uint8)
    else:
        print("Aviso: No se encontró máscara pulmonar específica, usando todo el corte.", flush=True)
        mask_lung_bin = np.ones((alto, ancho), dtype=np.uint8)
        
    # 3. Cargar máscara ROI
    if ruta_roi and os.path.exists(ruta_roi):
        print(f"Máscara ROI (Ground Truth): {os.path.basename(ruta_roi)}", flush=True)
        roi_sitk = sitk.ReadImage(ruta_roi)
        arr_roi = sitk.GetArrayFromImage(roi_sitk)[0]
        mask_roi_bin = (arr_roi > 0).astype(np.uint8)
    else:
        mask_roi_bin = np.zeros((alto, ancho), dtype=np.uint8)
        
    # 4. Enmascarar exterior a -1000 HU (corrección de bordes)
    matriz_enmascarada = matriz_img.copy()
    matriz_enmascarada[mask_lung_bin == 0] = -1000
    
    # 5. Recolectar parches válidos dentro del pulmón
    print("Filtrando parches pulmonares...", flush=True)
    parches_coords = []
    parches_arrays = []
    
    for y in range(0, alto - window_size + 1, stride):
        for x in range(0, ancho - window_size + 1, stride):
            parche_lung = mask_lung_bin[y:y+window_size, x:x+window_size]
            if np.mean(parche_lung) < min_lung_percent:
                continue
                
            parche_array = matriz_enmascarada[y:y+window_size, x:x+window_size]
            if np.mean(parche_array) < -980:
                continue
                
            parches_coords.append((y, x))
            parches_arrays.append(parche_array)
            
    num_parches = len(parches_coords)
    print(f"Total parches a evaluar: {num_parches} (Extracción paralela acelerada)...", flush=True)
    
    heatmap = np.zeros((alto, ancho), dtype=np.float32)
    conteo_ventanas = np.zeros((alto, ancho), dtype=np.float32)
    
    if num_parches > 0:
        features_list = joblib.Parallel(n_jobs=-1, batch_size=16)(
            joblib.delayed(extraer_caracteristicas_parche)(p, meta_features) for p in parches_arrays
        )
        
        X_df = pd.DataFrame(features_list, columns=top_features)
        probabilidades = rf_model.predict_proba(X_df)[:, 1]
        
        for (y, x), proba in zip(parches_coords, probabilidades):
            heatmap[y:y+window_size, x:x+window_size] += proba
            conteo_ventanas[y:y+window_size, x:x+window_size] += 1.0
            
    # 6. Promediar superposiciones, suavizar y restringir al pulmón
    conteo_ventanas[conteo_ventanas == 0] = 1.0
    heatmap_raw = (heatmap / conteo_ventanas)
    
    if num_parches > 0 and stride >= 8:
        heatmap_smooth = gaussian_filter(heatmap_raw, sigma=1.0)
    else:
        heatmap_smooth = heatmap_raw
        
    heatmap_final = heatmap_smooth * mask_lung_bin
    
    lung_pixels = int(np.sum(mask_lung_bin))
    fibrosis_pixels = int(np.sum((heatmap_final >= 0.5) & (mask_lung_bin == 1)))
    pct_fibrosis = (fibrosis_pixels / lung_pixels * 100.0) if lung_pixels > 0 else 0.0
    tiempo_total = time.time() - t_inicio
    
    print("\n" + "="*40, flush=True)
    print(f"RESULTADOS DEL CORTE (Tiempo: {tiempo_total:.1f} s)", flush=True)
    print("="*40, flush=True)
    print(f"Parches procesados:         {num_parches}", flush=True)
    print(f"Píxeles pulmonares:         {lung_pixels:,}", flush=True)
    print(f"Píxeles fibrosis (P>=0.5):  {fibrosis_pixels:,} ({pct_fibrosis:.2f}%)", flush=True)
    
    # 7. Visualización Clínica Comparativa de 3 Paneles
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    
    # Panel 1: TC Original
    axes[0].imshow(matriz_img, cmap='gray', vmin=-1000, vmax=400)
    axes[0].set_title(f"TC Pulmonar - Paciente {paciente_id} (Corte {idx_corte})")
    axes[0].axis('off')
    
    # Panel 2: Heatmap Radiómico
    axes[1].imshow(matriz_img, cmap='gray', vmin=-1000, vmax=400)
    im_heat = axes[1].imshow(heatmap_final, cmap='jet', alpha=0.45, vmin=0, vmax=1)
    axes[1].set_title(f"Mapa Radiomico (Fibrosis: {pct_fibrosis:.1f}%)")
    axes[1].axis('off')
    cbar = fig.colorbar(im_heat, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label("P(Fibrosis)")
    
    # Panel 3: Ground Truth
    axes[2].imshow(matriz_img, cmap='gray', vmin=-1000, vmax=400)
    if np.sum(mask_roi_bin) > 0:
        axes[2].imshow(mask_roi_bin, cmap='autumn', alpha=0.5)
        axes[2].set_title("Ground Truth (Fibrosis Anotada)")
    else:
        axes[2].set_title("Ground Truth (Sin patologia anotada)")
    axes[2].axis('off')
    
    plt.tight_layout()
    
    archivo_salida_img = os.path.join(DATA_DIR, f"mapa_color_2d_paciente_{paciente_id}_corte{idx_corte}.png")
    plt.savefig(archivo_salida_img, dpi=300, bbox_inches='tight')
    print(f"Figura guardada en: {archivo_salida_img}", flush=True)
    plt.close()
    
    return heatmap_final, pct_fibrosis


if __name__ == "__main__":
    generar_mapa_2d()