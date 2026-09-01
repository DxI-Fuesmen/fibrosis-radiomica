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
from matplotlib.widgets import Slider
from scipy.ndimage import gaussian_filter
from radiomics import featureextractor
import radiomics

import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from segmentacion_vasos_vias import preparar_tc_libre_de_vasos, to_hu

warnings.filterwarnings("ignore")
radiomics.setVerbosity(logging.ERROR)
logging.getLogger("radiomics").setLevel(logging.ERROR)

# ==========================================
# 1. CONFIGURACIÓN GENERAL Y RUTAS
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")

BASE_REPO = os.path.join(BASE_DIR, "..", "base")
BASE_ILD = os.path.join(BASE_REPO, "ILD_DB")
DIR_CLASIFICADA = os.path.join(BASE_ILD, "ILD_DB_Clasificada")
DIR_LUNGMASKS = os.path.join(BASE_ILD, "ILD_DB_lungMasks")

PACIENTE_DEFAULT = "77"
CATEGORIA_DEFAULT = "fibrosis"

WINDOW_SIZE = 24  # Alta resolución espacial
STRIDE = 8
MIN_LUNG_PERCENT = 0.35

ARCHIVO_MODELO_MULTI = os.path.join(MODELS_DIR, "mejor_modelo_multiclase.pkl")


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


def extraer_numero_corte(nombre_archivo):
    match = re.search(r'(\d+)\.dcm$', nombre_archivo, re.IGNORECASE)
    return int(match.group(1)) if match else 0


def recopilar_serie_paciente(paciente_id=PACIENTE_DEFAULT, categoria=CATEGORIA_DEFAULT):
    dir_ct = os.path.join(DIR_CLASIFICADA, categoria, str(paciente_id))
    dir_lung = os.path.join(DIR_LUNGMASKS, str(paciente_id), "lung_mask")
    dir_roi = os.path.join(DIR_CLASIFICADA, categoria, str(paciente_id), "roi_mask")
    
    if not os.path.exists(dir_ct):
        dir_ct = os.path.join(DIR_LUNGMASKS, str(paciente_id))
        
    if not os.path.exists(dir_ct):
        raise FileNotFoundError(f"No se encontró la carpeta del paciente {paciente_id} en: {dir_ct}")
        
    archivos_ct = glob.glob(os.path.join(dir_ct, "CT-*.dcm"))
    if not archivos_ct:
        archivos_ct = [f for f in glob.glob(os.path.join(dir_ct, "*.dcm")) if "mask" not in os.path.basename(f).lower()]
        
    archivos_ct.sort(key=extraer_numero_corte)
    
    serie = []
    for f_ct in archivos_ct:
        idx = extraer_numero_corte(f_ct)
        nombre_base = os.path.basename(f_ct)
        
        f_lung_pattern = os.path.join(dir_lung, f"lung_mask_*_{idx}.dcm")
        match_lung = glob.glob(f_lung_pattern)
        f_lung = match_lung[0] if match_lung else None
        
        f_roi_pattern = os.path.join(dir_roi, f"roi_mask_*_{idx}.dcm")
        match_roi = glob.glob(f_roi_pattern)
        f_roi = match_roi[0] if match_roi else None
        
        serie.append({
            'slice_index': idx,
            'filename': nombre_base,
            'ct_path': f_ct,
            'lung_mask_path': f_lung,
            'roi_mask_path': f_roi
        })
        
    return serie


def procesar_volumen_paciente(paciente_id=PACIENTE_DEFAULT, categoria=CATEGORIA_DEFAULT, stride=STRIDE, window_size=WINDOW_SIZE, min_lung_percent=MIN_LUNG_PERCENT):
    t_inicio_global = time.time()
    print("="*60, flush=True)
    print(f"ANÁLISIS RADIÓMICO VOLUMÉTRICO 3D MULTICLASE (LIBRE DE VASOS) - PACIENTE {paciente_id}", flush=True)
    print(f"Salto (Stride): {stride} px | Ventana: {window_size}x{window_size} px", flush=True)
    print("="*60, flush=True)
    
    artefacto = cargar_modelo_multiclase()
    clf = artefacto['model']
    scaler = artefacto['scaler']
    top_features = artefacto['features']
    
    serie = recopilar_serie_paciente(paciente_id, categoria)
    num_cortes = len(serie)
    print(f"Cortes encontrados para procesar: {num_cortes}\n", flush=True)
    
    if num_cortes == 0:
        raise ValueError(f"No se encontraron cortes DICOM para el paciente {paciente_id}")
        
    vol_ct_hu = []
    vol_lung_mask = []
    vol_roi_mask = []
    vol_proba_sano = []
    vol_proba_ggo = []
    vol_proba_fib = []
    slice_metrics = []
    
    total_lung_voxels = 0
    total_fib_voxels = 0
    total_ggo_voxels = 0
    total_sano_voxels = 0
    total_roi_voxels = 0
    total_vasos_voxels = 0
    
    for i, item in enumerate(serie):
        t_slice = time.time()
        idx = item['slice_index']
        print(f"--> Procesando Corte [{i+1:02d}/{num_cortes:02d}] - {item['filename']}...", flush=True)
        
        # 1. Cargar corte CT
        img_sitk = sitk.ReadImage(item['ct_path'])
        arr_ct_raw = sitk.GetArrayFromImage(img_sitk)[0]
        alto, ancho = arr_ct_raw.shape
        
        # 2. Cargar máscara pulmonar
        if item['lung_mask_path'] and os.path.exists(item['lung_mask_path']):
            lung_sitk = sitk.ReadImage(item['lung_mask_path'])
            arr_lung = sitk.GetArrayFromImage(lung_sitk)[0]
            arr_lung_bin = (arr_lung > 0).astype(np.uint8)
        else:
            arr_lung_bin = np.ones((alto, ancho), dtype=np.uint8)
            
        # 3. Cargar máscara ROI
        if item['roi_mask_path'] and os.path.exists(item['roi_mask_path']):
            roi_sitk = sitk.ReadImage(item['roi_mask_path'])
            arr_roi = sitk.GetArrayFromImage(roi_sitk)[0]
            arr_roi_bin = (arr_roi > 0).astype(np.uint8)
        else:
            arr_roi_bin = np.zeros((alto, ancho), dtype=np.uint8)
            
        lung_pixels_corte = int(np.sum(arr_lung_bin))
        roi_pixels_corte = int(np.sum(arr_roi_bin))
        
        total_lung_voxels += lung_pixels_corte
        total_roi_voxels += roi_pixels_corte
        
        if lung_pixels_corte == 0:
            vol_ct_hu.append(to_hu(arr_ct_raw))
            vol_lung_mask.append(arr_lung_bin)
            vol_roi_mask.append(arr_roi_bin)
            vol_proba_sano.append(np.zeros((alto, ancho), dtype=np.float32))
            vol_proba_ggo.append(np.zeros((alto, ancho), dtype=np.float32))
            vol_proba_fib.append(np.zeros((alto, ancho), dtype=np.float32))
            slice_metrics.append({
                'slice': idx, 'filename': item['filename'],
                'lung_pixels': 0, 'sano_pct': 0.0, 'ggo_pct': 0.0, 'fibrosis_pct': 0.0,
                'roi_pixels': roi_pixels_corte
            })
            continue
            
        # 4. EXCLUSIÓN VASCULAR E INPAINTING
        arr_libre, mask_vasos, arr_hu = preparar_tc_libre_de_vasos(arr_ct_raw, arr_lung_bin)
        vasos_px_corte = int(np.sum(mask_vasos & (arr_lung_bin == 1)))
        total_vasos_voxels += vasos_px_corte
        
        # 5. Filtrar parches válidos
        parches_coords = []
        parches_arrays = []
        
        for y in range(0, alto - window_size + 1, stride):
            for x in range(0, ancho - window_size + 1, stride):
                p_l = arr_lung_bin[y:y+window_size, x:x+window_size]
                if np.mean(p_l) < min_lung_percent:
                    continue
                p_c = arr_libre[y:y+window_size, x:x+window_size]
                if np.mean(p_c) < -980:
                    continue
                parches_coords.append((y, x))
                parches_arrays.append(p_c)
                
        num_parches = len(parches_coords)
        hm_s = np.zeros((alto, ancho), dtype=np.float32)
        hm_g = np.zeros((alto, ancho), dtype=np.float32)
        hm_f = np.zeros((alto, ancho), dtype=np.float32)
        conteo_acc = np.zeros((alto, ancho), dtype=np.float32)
        
        if num_parches > 0:
            features_list = joblib.Parallel(n_jobs=-1, batch_size=16)(
                joblib.delayed(extraer_caracteristicas_parche)(p, artefacto) for p in parches_arrays
            )
            
            X_df = pd.DataFrame(features_list, columns=top_features)
            X_sc = scaler.transform(X_df.values)
            probas = clf.predict_proba(X_sc)
            
            for (y, x), pr in zip(parches_coords, probas):
                hm_s[y:y+window_size, x:x+window_size] += pr[0]
                hm_g[y:y+window_size, x:x+window_size] += pr[1]
                hm_f[y:y+window_size, x:x+window_size] += pr[2]
                conteo_acc[y:y+window_size, x:x+window_size] += 1.0
                
        conteo_acc[conteo_acc == 0] = 1.0
        
        hm_s_sm = gaussian_filter(hm_s / conteo_acc, sigma=1.0) * arr_lung_bin
        hm_g_sm = gaussian_filter(hm_g / conteo_acc, sigma=1.0) * arr_lung_bin
        hm_f_sm = gaussian_filter(hm_f / conteo_acc, sigma=1.0) * arr_lung_bin
        
        # Calibración clínica
        es_f = (hm_f_sm >= 0.45) & (hm_f_sm >= hm_g_sm) & (arr_lung_bin == 1)
        es_g = (hm_g_sm >= 0.50) & (~es_f) & (arr_lung_bin == 1)
        es_s = (~es_f) & (~es_g) & (arr_lung_bin == 1)
        
        s_px = int(np.sum(es_s))
        g_px = int(np.sum(es_g))
        f_px = int(np.sum(es_f))
        
        total_sano_voxels += s_px
        total_ggo_voxels += g_px
        total_fib_voxels += f_px
        
        pct_s = (s_px / lung_pixels_corte * 100.0) if lung_pixels_corte > 0 else 0.0
        pct_g = (g_px / lung_pixels_corte * 100.0) if lung_pixels_corte > 0 else 0.0
        pct_f = (f_px / lung_pixels_corte * 100.0) if lung_pixels_corte > 0 else 0.0
        
        dt_corte = time.time() - t_slice
        print(f"    Parches: {num_parches:3d} | Sano: {pct_s:4.1f}% | GGO: {pct_g:4.1f}% | Fib: {pct_f:4.1f}% | T: {dt_corte:4.1f}s", flush=True)
        
        vol_ct_hu.append(arr_hu)
        vol_lung_mask.append(arr_lung_bin)
        vol_roi_mask.append(arr_roi_bin)
        vol_proba_sano.append(hm_s_sm)
        vol_proba_ggo.append(hm_g_sm)
        vol_proba_fib.append(hm_f_sm)
        
        slice_metrics.append({
            'slice': idx, 'filename': item['filename'],
            'lung_pixels': lung_pixels_corte,
            'sano_pct': pct_s, 'ggo_pct': pct_g, 'fibrosis_pct': pct_f,
            'roi_pixels': roi_pixels_corte
        })
        
    vol_ct_hu = np.array(vol_ct_hu)
    vol_lung_mask = np.array(vol_lung_mask)
    vol_roi_mask = np.array(vol_roi_mask)
    vol_proba_sano = np.array(vol_proba_sano)
    vol_proba_ggo = np.array(vol_proba_ggo)
    vol_proba_fib = np.array(vol_proba_fib)
    
    pct_global_sano = (total_sano_voxels / total_lung_voxels * 100.0) if total_lung_voxels > 0 else 0.0
    pct_global_ggo = (total_ggo_voxels / total_lung_voxels * 100.0) if total_lung_voxels > 0 else 0.0
    pct_global_fib = (total_fib_voxels / total_lung_voxels * 100.0) if total_lung_voxels > 0 else 0.0
    pct_global_roi = (total_roi_voxels / total_lung_voxels * 100.0) if total_lung_voxels > 0 else 0.0
    pct_global_vasos = (total_vasos_voxels / total_lung_voxels * 100.0) if total_lung_voxels > 0 else 0.0
    
    tiempo_total_global = time.time() - t_inicio_global
    
    print("\n" + "="*60, flush=True)
    print("RESUMEN CLÍNICO VOLUMÉTRICO MULTICLASE (LIBRE DE VASOS)", flush=True)
    print("="*60, flush=True)
    print(f"Paciente:                          {paciente_id}", flush=True)
    print(f"Cortes Procesados:                 {num_cortes} (Tiempo: {tiempo_total_global:.1f} s)", flush=True)
    print(f"Volumen Pulmonar Total:            {total_lung_voxels:,} voxels", flush=True)
    print(f"  -> Vasos Excluidos:              {total_vasos_voxels:,} voxels ({pct_global_vasos:.1f}%)", flush=True)
    print(f"  -> Parénquima Sano:              {total_sano_voxels:,} voxels ({pct_global_sano:.2f}%)", flush=True)
    print(f"  -> Vidrio Esmerilado (GGO):      {total_ggo_voxels:,} voxels ({pct_global_ggo:.2f}%)", flush=True)
    print(f"  -> Fibrosis Establecida:         {total_fib_voxels:,} voxels ({pct_global_fib:.2f}%)", flush=True)
    if total_roi_voxels > 0:
        print(f"Carga Real de Fibrosis (Ground Truth): {pct_global_roi:.2f}%", flush=True)
    print("="*60 + "\n", flush=True)
    
    archivo_salida_npz = os.path.join(DATA_DIR, f"heatmap_volumen_paciente_{paciente_id}.npz")
    np.savez_compressed(
        archivo_salida_npz,
        vol_ct=vol_ct_hu,
        vol_lung_mask=vol_lung_mask,
        vol_roi_mask=vol_roi_mask,
        vol_proba_sano=vol_proba_sano,
        vol_proba_ggo=vol_proba_ggo,
        vol_proba_fib=vol_proba_fib,
        pct_global_sano=pct_global_sano,
        pct_global_ggo=pct_global_ggo,
        pct_global_fib=pct_global_fib,
        slice_metrics=json.dumps(slice_metrics)
    )
    print(f"Volumen 3D guardado exitosamente en: {archivo_salida_npz}", flush=True)
    
    guardar_resumen_estatico_multiclase(vol_ct_hu, vol_proba_fib, vol_proba_ggo, vol_roi_mask, slice_metrics, paciente_id, pct_global_fib, pct_global_ggo)
    
    return vol_ct_hu, vol_lung_mask, vol_roi_mask, vol_proba_fib, vol_proba_ggo, slice_metrics, pct_global_fib


def guardar_resumen_estatico_multiclase(vol_ct_hu, vol_proba_fib, vol_proba_ggo, vol_roi_mask, slice_metrics, paciente_id, pct_fib, pct_ggo):
    valid_indices = [i for i, m in enumerate(slice_metrics) if m['lung_pixels'] > 0]
    if not valid_indices:
        return
        
    step = max(1, len(valid_indices) // 6)
    selected_indices = valid_indices[::step][:6]
    
    n_sel = len(selected_indices)
    fig, axes = plt.subplots(n_sel, 3, figsize=(15, 4 * n_sel))
    if n_sel == 1:
        axes = np.expand_dims(axes, 0)
        
    for row, idx in enumerate(selected_indices):
        m = slice_metrics[idx]
        
        # TC
        axes[row, 0].imshow(vol_ct_hu[idx], cmap='gray', vmin=-1000, vmax=400)
        axes[row, 0].set_title(f"Corte {m['slice']} ({m['filename']}) - TC")
        axes[row, 0].axis('off')
        
        # Mapa Tri-Color
        axes[row, 1].imshow(vol_ct_hu[idx], cmap='gray', vmin=-1000, vmax=400)
        overlay = np.zeros((*vol_ct_hu[idx].shape, 4))
        p_f = vol_proba_fib[idx]
        p_g = vol_proba_ggo[idx]
        overlay[(p_f >= 0.45) & (p_f >= p_g)] = [1.0, 0.1, 0.1, 0.65]  # Rojo Fibrosis
        overlay[(p_g >= 0.50) & (p_f < 0.45)] = [1.0, 0.85, 0.0, 0.65]  # Amarillo GGO
        axes[row, 1].imshow(overlay)
        axes[row, 1].set_title(f"Tri-Color (Fib: {m['fibrosis_pct']:.1f}% | GGO: {m['ggo_pct']:.1f}%)")
        axes[row, 1].axis('off')
        
        # Ground Truth
        axes[row, 2].imshow(vol_ct_hu[idx], cmap='gray', vmin=-1000, vmax=400)
        if np.sum(vol_roi_mask[idx]) > 0:
            axes[row, 2].imshow(vol_roi_mask[idx], cmap='autumn', alpha=0.5)
            axes[row, 2].set_title("Ground Truth (Anotación Médica)")
        else:
            axes[row, 2].set_title("Ground Truth (Sin patología)")
        axes[row, 2].axis('off')
        
    plt.suptitle(f"Análisis Volumétrico Libre de Vasos - Paciente {paciente_id} (Fibrosis: {pct_fib:.1f}% | GGO: {pct_ggo:.1f}%)", fontsize=16)
    plt.tight_layout()
    
    archivo_resumen = os.path.join(DATA_DIR, f"resumen_volumen_paciente_{paciente_id}.png")
    plt.savefig(archivo_resumen, dpi=200, bbox_inches='tight')
    print(f"Resumen visual guardado en: {archivo_resumen}", flush=True)
    plt.close()


def visor_interactivo_3d(vol_ct_hu, vol_proba_fib, vol_proba_ggo, vol_roi_mask, slice_metrics, paciente_id, pct_fib, pct_ggo):
    num_cortes = len(vol_ct_hu)
    fig, axes = plt.subplots(1, 3, figsize=(16, 7))
    plt.subplots_adjust(bottom=0.20)
    
    corte_inicial = min(num_cortes - 1, max(0, num_cortes // 2))
    
    ax_slider = plt.axes([0.25, 0.08, 0.50, 0.04])
    slider = Slider(ax_slider, 'Corte Axial', 0, num_cortes - 1, valinit=corte_inicial, valstep=1)
    
    def update(val):
        idx = int(slider.val)
        m = slice_metrics[idx]
        
        axes[0].clear()
        axes[0].imshow(vol_ct_hu[idx], cmap='gray', vmin=-1000, vmax=400)
        axes[0].set_title(f"TC: {m['filename']} (Corte {m['slice']})")
        axes[0].axis('off')
        
        axes[1].clear()
        axes[1].imshow(vol_ct_hu[idx], cmap='gray', vmin=-1000, vmax=400)
        overlay = np.zeros((*vol_ct_hu[idx].shape, 4))
        p_f = vol_proba_fib[idx]
        p_g = vol_proba_ggo[idx]
        overlay[(p_f >= 0.45) & (p_f >= p_g)] = [1.0, 0.1, 0.1, 0.65]
        overlay[(p_g >= 0.50) & (p_f < 0.45)] = [1.0, 0.85, 0.0, 0.65]
        axes[1].imshow(overlay)
        axes[1].set_title(f"Tri-Color (Fib: {m['fibrosis_pct']:.1f}% | GGO: {m['ggo_pct']:.1f}%)")
        axes[1].axis('off')
        
        axes[2].clear()
        axes[2].imshow(vol_ct_hu[idx], cmap='gray', vmin=-1000, vmax=400)
        if np.sum(vol_roi_mask[idx]) > 0:
            axes[2].imshow(vol_roi_mask[idx], cmap='autumn', alpha=0.5)
            axes[2].set_title("Ground Truth")
        else:
            axes[2].set_title("Ground Truth (Sin patología)")
        axes[2].axis('off')
        
        fig.canvas.draw_idle()
        
    slider.on_changed(update)
    update(corte_inicial)
    
    plt.suptitle(f"Explorador Volumétrico 3D Libre de Vasos - Paciente {paciente_id} | Fibrosis: {pct_fib:.1f}% | GGO: {pct_ggo:.1f}%)", fontsize=15)
    plt.show()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Análisis Volumétrico 3D Libre de Vasos")
    parser.add_argument("--paciente", type=str, default=PACIENTE_DEFAULT, help="ID del paciente (ej: 77)")
    parser.add_argument("--categoria", type=str, default=CATEGORIA_DEFAULT, help="Categoría (fibrosis o healthy)")
    parser.add_argument("--stride", type=int, default=STRIDE, help="Salto en píxeles (default: 8)")
    parser.add_argument("--min_lung", type=float, default=MIN_LUNG_PERCENT, help="Porcentaje mínimo de pulmón (default: 0.35)")
    parser.add_argument("--interactive", action="store_true", help="Abrir visor interactivo con slider")
    args = parser.parse_args()
    
    vol_ct_hu, vol_lung_mask, vol_roi_mask, vol_proba_fib, vol_proba_ggo, slice_metrics, pct_fib = procesar_volumen_paciente(
        paciente_id=args.paciente,
        categoria=args.categoria,
        stride=args.stride,
        window_size=WINDOW_SIZE,
        min_lung_percent=args.min_lung
    )
    
    if args.interactive:
        visor_interactivo_3d(vol_ct_hu, vol_proba_fib, vol_proba_ggo, vol_roi_mask, slice_metrics, args.paciente, pct_fib, 0.0)


if __name__ == "__main__":
    main()
