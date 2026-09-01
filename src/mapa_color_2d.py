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
import sys
from scipy.ndimage import gaussian_filter
from radiomics import featureextractor
import radiomics

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from segmentacion_vasos_vias import preparar_tc_libre_de_vasos, to_hu

warnings.filterwarnings("ignore")
radiomics.setVerbosity(logging.ERROR)
logging.getLogger("radiomics").setLevel(logging.ERROR)

# ==========================================
# 1. CONFIGURACIÓN Y RUTAS
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")

BASE_REPO = os.path.join(BASE_DIR, "..", "base")
BASE_ILD = os.path.join(BASE_REPO, "ILD_DB")
DIR_CLASIFICADA = os.path.join(BASE_ILD, "ILD_DB_Clasificada")
RUTA_DICOM_DEFAULT = os.path.join(DIR_CLASIFICADA, "fibrosis", "77", "CT-0002-0012.dcm")

ARCHIVO_MODELO_MULTI = os.path.join(MODELS_DIR, "mejor_modelo_multiclase.pkl")

WINDOW_SIZE = 24  # Alta resolución espacial (16 mm x 16 mm)
STRIDE = 8       # Salto de 8 px para mapas de alta densidad
MIN_LUNG_PERCENT = 0.35


def cargar_modelo_multiclase():
    if not os.path.exists(ARCHIVO_MODELO_MULTI):
        raise FileNotFoundError(f"No se encontró el modelo en: {ARCHIVO_MODELO_MULTI}")
    artefacto = joblib.load(ARCHIVO_MODELO_MULTI)
    return artefacto


def extraer_caracteristicas_parche(parche_array, meta_features):
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.disableAllImageTypes()
    extractor.disableAllFeatures()
    
    for img_type in meta_features.get('required_image_types', []):
        if img_type.lower() == 'log':
            extractor.enableImageTypeByName('LoG', customArgs={'sigma': [1.0, 2.0, 3.0, 5.0]})
        else:
            extractor.enableImageTypeByName(img_type)
            
    for feat_cls in ['firstorder', 'glcm', 'glrlm', 'glszm', 'gldm', 'ngtdm']:
        extractor.enableFeatureClassByName(feat_cls)
        
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
    nombre_archivo = os.path.basename(ruta_dicom)
    dir_paciente = os.path.dirname(ruta_dicom)
    
    match_corte = re.search(r'(\d+)\.dcm$', nombre_archivo, re.IGNORECASE)
    idx_corte = int(match_corte.group(1)) if match_corte else 1
    
    partes = ruta_dicom.replace('\\', '/').split('/')
    paciente_id = partes[-2] if len(partes) >= 2 else "77"
    
    dir_lung = os.path.join(BASE_ILD, "ILD_DB_lungMasks", paciente_id, "lung_mask")
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
    print(f"ANÁLISIS RADIÓMICO 2D DE ALTA RESOLUCIÓN (LIBRE DE VASOS)", flush=True)
    print(f"Corte: {os.path.basename(ruta_dicom)}", flush=True)
    print(f"Salto (Stride): {stride} px | Ventana: {window_size}x{window_size} px", flush=True)
    print("="*60, flush=True)
    
    artefacto = cargar_modelo_multiclase()
    clf = artefacto['model']
    scaler = artefacto['scaler']
    top_features = artefacto['features']
    
    ruta_lung, ruta_roi, idx_corte, paciente_id = buscar_mascaras_asociadas(ruta_dicom)
    
    # 1. Cargar imagen CT
    img_sitk = sitk.ReadImage(ruta_dicom)
    matriz_raw = sitk.GetArrayFromImage(img_sitk)[0]
    alto, ancho = matriz_raw.shape
    
    # 2. Cargar máscara pulmonar
    if ruta_lung and os.path.exists(ruta_lung):
        lung_sitk = sitk.ReadImage(ruta_lung)
        arr_lung = sitk.GetArrayFromImage(lung_sitk)[0]
        mask_lung_bin = (arr_lung > 0).astype(np.uint8)
    else:
        mask_lung_bin = np.ones((alto, ancho), dtype=np.uint8)
        
    # 3. Cargar máscara ROI (Ground Truth)
    if ruta_roi and os.path.exists(ruta_roi):
        roi_sitk = sitk.ReadImage(ruta_roi)
        arr_roi = sitk.GetArrayFromImage(roi_sitk)[0]
        mask_roi_bin = (arr_roi > 0).astype(np.uint8)
    else:
        mask_roi_bin = np.zeros((alto, ancho), dtype=np.uint8)
        
    # 4. EXCLUSIÓN VASCULAR E INPAINTING
    matriz_libre, mask_vasos, matriz_hu = preparar_tc_libre_de_vasos(matriz_raw, mask_lung_bin)
    
    # 5. Recolectar parches de parénquima puro
    parches_coords = []
    parches_arrays = []
    
    for y in range(0, alto - window_size + 1, stride):
        for x in range(0, ancho - window_size + 1, stride):
            p_l = mask_lung_bin[y:y+window_size, x:x+window_size]
            if np.mean(p_l) < min_lung_percent:
                continue
                
            p_ct = matriz_libre[y:y+window_size, x:x+window_size]
            if np.mean(p_ct) < -980:
                continue
                
            parches_coords.append((y, x))
            parches_arrays.append(p_ct)
            
    num_parches = len(parches_coords)
    print(f"Total parches a evaluar: {num_parches} (Extracción paralela acelerada)...", flush=True)
    
    hm_sano = np.zeros((alto, ancho), dtype=np.float32)
    hm_ggo = np.zeros((alto, ancho), dtype=np.float32)
    hm_fib = np.zeros((alto, ancho), dtype=np.float32)
    conteo_ventanas = np.zeros((alto, ancho), dtype=np.float32)
    
    if num_parches > 0:
        features_list = joblib.Parallel(n_jobs=-1, batch_size=16)(
            joblib.delayed(extraer_caracteristicas_parche)(p, artefacto) for p in parches_arrays
        )
        
        X_df = pd.DataFrame(features_list, columns=top_features)
        X_sc = scaler.transform(X_df.values)
        probas = clf.predict_proba(X_sc)
        
        for (y, x), pr in zip(parches_coords, probas):
            hm_sano[y:y+window_size, x:x+window_size] += pr[0]
            hm_ggo[y:y+window_size, x:x+window_size] += pr[1]
            hm_fib[y:y+window_size, x:x+window_size] += pr[2]
            conteo_ventanas[y:y+window_size, x:x+window_size] += 1.0
            
    conteo_ventanas[conteo_ventanas == 0] = 1.0
    
    # Suavizado gaussiano
    hm_sano_sm = gaussian_filter(hm_sano / conteo_ventanas, sigma=1.0) * mask_lung_bin
    hm_ggo_sm = gaussian_filter(hm_ggo / conteo_ventanas, sigma=1.0) * mask_lung_bin
    hm_fib_sm = gaussian_filter(hm_fib / conteo_ventanas, sigma=1.0) * mask_lung_bin
    
    # Calibración clínica
    es_fib = (hm_fib_sm >= 0.45) & (hm_fib_sm >= hm_ggo_sm) & (mask_lung_bin == 1)
    es_ggo = (hm_ggo_sm >= 0.50) & (~es_fib) & (mask_lung_bin == 1)
    es_sano = (~es_fib) & (~es_ggo) & (mask_lung_bin == 1)
    
    lung_pixels = int(np.sum(mask_lung_bin))
    sano_px = int(np.sum(es_sano))
    ggo_px = int(np.sum(es_ggo))
    fib_px = int(np.sum(es_fib))
    vasos_px = int(np.sum(mask_vasos & (mask_lung_bin == 1)))
    
    pct_sano = (sano_px / lung_pixels * 100.0) if lung_pixels > 0 else 0.0
    pct_ggo = (ggo_px / lung_pixels * 100.0) if lung_pixels > 0 else 0.0
    pct_fib = (fib_px / lung_pixels * 100.0) if lung_pixels > 0 else 0.0
    tiempo_total = time.time() - t_inicio
    
    print("\n" + "="*40, flush=True)
    print(f"RESULTADOS DEL CORTE (Tiempo: {tiempo_total:.1f} s)", flush=True)
    print("="*40, flush=True)
    print(f"Píxeles pulmonares: {lung_pixels:,}")
    print(f"  -> Vasos excluidos: {vasos_px:,} ({vasos_px/lung_pixels*100:.1f}%)")
    print(f"  -> Sano:            {sano_px:,} ({pct_sano:.1f}%)")
    print(f"  -> Vidrio E. (GGO): {ggo_px:,} ({pct_ggo:.1f}%)")
    print(f"  -> Fibrosis:        {fib_px:,} ({pct_fib:.1f}%)")
    
    # Visualización de 3 Paneles
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    
    axes[0].imshow(matriz_hu, cmap='gray', vmin=-1000, vmax=400)
    axes[0].set_title(f"TC Pulmonar - Paciente {paciente_id} (Corte {idx_corte})")
    axes[0].axis('off')
    
    # Mapa Tri-Color (Rojo: Fibrosis, Amarillo: GGO)
    axes[1].imshow(matriz_hu, cmap='gray', vmin=-1000, vmax=400)
    overlay = np.zeros((*matriz_hu.shape, 4))
    overlay[es_fib] = [1.0, 0.1, 0.1, 0.65]  # Rojo Fibrosis
    overlay[es_ggo] = [1.0, 0.85, 0.0, 0.65] # Amarillo GGO
    axes[1].imshow(overlay)
    axes[1].set_title(f"Mapa Tri-Color (Fib: {pct_fib:.1f}% | GGO: {pct_ggo:.1f}%)")
    axes[1].axis('off')
    
    # Ground Truth
    axes[2].imshow(matriz_hu, cmap='gray', vmin=-1000, vmax=400)
    if np.sum(mask_roi_bin) > 0:
        axes[2].imshow(mask_roi_bin, cmap='autumn', alpha=0.5)
        axes[2].set_title("Ground Truth (Anotación Médica)")
    else:
        axes[2].set_title("Ground Truth (Sin patología)")
    axes[2].axis('off')
    
    plt.tight_layout()
    archivo_salida_img = os.path.join(DATA_DIR, f"mapa_color_2d_paciente_{paciente_id}_corte{idx_corte}.png")
    plt.savefig(archivo_salida_img, dpi=300, bbox_inches='tight')
    print(f"Figura guardada en: {archivo_salida_img}", flush=True)
    plt.close()


if __name__ == "__main__":
    generar_mapa_2d()