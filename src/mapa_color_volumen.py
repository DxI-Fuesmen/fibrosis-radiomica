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

warnings.filterwarnings("ignore")
radiomics.setVerbosity(logging.ERROR)
logging.getLogger("radiomics").setLevel(logging.ERROR)

# ==========================================
# 1. CONFIGURACIÓN GENERAL Y RUTAS
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")

# Rutas de la base de datos ILD_DB
BASE_ILD = r"C:\Users\malcalde\Documents\Base Pulmon\ILD_DB"
DIR_CLASIFICADA = os.path.join(BASE_ILD, "ILD_DB_Clasificada")
DIR_LUNGMASKS = os.path.join(BASE_ILD, "ILD_DB_lungMasks")

# Paciente por defecto para análisis
PACIENTE_DEFAULT = "77"
CATEGORIA_DEFAULT = "fibrosis"

# Parámetros de la ventana deslizante
WINDOW_SIZE = 32
STRIDE = 8  # Salto de 8 píxeles (óptimo para velocidad y alta resolución)
MIN_LUNG_PERCENT = 0.40  # Requiere al menos 40% de pulmón en el parche para evitar artefactos de borde

# Archivos del modelo y características
ARCHIVO_MODELO = os.path.join(MODELS_DIR, "modelo_rf_fibrosis.pkl")
ARCHIVO_JSON_FEAT = os.path.join(DATA_DIR, "selected_features.json")


def cargar_modelo_y_features():
    """Carga el modelo entrenado y la lista ordenada de características seleccionadas."""
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


def extraer_numero_corte(nombre_archivo):
    """Extrae el índice numérico del corte a partir del nombre de archivo."""
    match = re.search(r'(\d+)\.dcm$', nombre_archivo, re.IGNORECASE)
    return int(match.group(1)) if match else 0


def recopilar_serie_paciente(paciente_id=PACIENTE_DEFAULT, categoria=CATEGORIA_DEFAULT):
    """
    Empareja los cortes DICOM del paciente con sus respectivas máscaras de pulmón y ROI.
    """
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
    """
    Ejecuta el análisis radiómico volumétrico 3D corrigiendo el artefacto de borde mediante enmascaramiento externo a -1000 HU.
    """
    t_inicio_global = time.time()
    print("="*60, flush=True)
    print(f"ANÁLISIS RADIÓMICO VOLUMÉTRICO 3D - PACIENTE {paciente_id}", flush=True)
    print(f"Salto (Stride): {stride} px | Ventana: {window_size}x{window_size} px | Umbral Pulmón: {min_lung_percent*100:.0f}%", flush=True)
    print("="*60, flush=True)
    
    rf_model, meta_features = cargar_modelo_y_features()
    top_features = meta_features['features']
    
    serie = recopilar_serie_paciente(paciente_id, categoria)
    num_cortes = len(serie)
    print(f"Cortes encontrados para procesar: {num_cortes}\n", flush=True)
    
    if num_cortes == 0:
        raise ValueError(f"No se encontraron cortes DICOM para el paciente {paciente_id}")
        
    vol_ct = []
    vol_lung_mask = []
    vol_roi_mask = []
    vol_heatmaps = []
    slice_metrics = []
    
    total_lung_voxels = 0
    total_fibrosis_voxels = 0
    total_roi_voxels = 0
    
    for i, item in enumerate(serie):
        t_slice = time.time()
        idx = item['slice_index']
        print(f"--> Procesando Corte [{i+1:02d}/{num_cortes:02d}] - {item['filename']}...", flush=True)
        
        # 1. Cargar corte CT
        img_sitk = sitk.ReadImage(item['ct_path'])
        arr_ct = sitk.GetArrayFromImage(img_sitk)[0]  # (H, W)
        alto, ancho = arr_ct.shape
        
        # 2. Cargar máscara pulmonar
        if item['lung_mask_path'] and os.path.exists(item['lung_mask_path']):
            lung_sitk = sitk.ReadImage(item['lung_mask_path'])
            arr_lung = sitk.GetArrayFromImage(lung_sitk)[0]
            arr_lung_bin = (arr_lung > 0).astype(np.uint8)
        else:
            arr_lung_bin = np.ones((alto, ancho), dtype=np.uint8)
            
        # 3. Cargar máscara ROI (Ground Truth) si existe
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
        
        # Si el corte no tiene tejido pulmonar, omitir
        if lung_pixels_corte == 0:
            print(f"    (Sin parenquima pulmonar detectado, corte omitido en {time.time()-t_slice:.2f}s)", flush=True)
            heatmap_corte = np.zeros((alto, ancho), dtype=np.float32)
            vol_ct.append(arr_ct)
            vol_lung_mask.append(arr_lung_bin)
            vol_roi_mask.append(arr_roi_bin)
            vol_heatmaps.append(heatmap_corte)
            slice_metrics.append({
                'slice': idx,
                'filename': item['filename'],
                'lung_pixels': 0,
                'fibrosis_pixels': 0,
                'fibrosis_pct': 0.0,
                'roi_pixels': roi_pixels_corte
            })
            continue
            
        # 4. CORRECCIÓN CLAVE DE BORDES:
        # Enmascarar la imagen TC asignando -1000 HU (aire) a todo lo que esté fuera del pulmón.
        # Esto elimina el salto artificial de densidad (+50 HU a +800 HU de costillas/mediastino)
        # que causaba que los bordes normales se clasificaran como fibrosis.
        arr_ct_enmascarado = arr_ct.copy()
        arr_ct_enmascarado[arr_lung_bin == 0] = -1000
        
        # 5. Filtrar parches válidos en el corte
        parches_coords = []
        parches_arrays = []
        
        for y in range(0, alto - window_size + 1, stride):
            for x in range(0, ancho - window_size + 1, stride):
                parche_lung = arr_lung_bin[y:y+window_size, x:x+window_size]
                if np.mean(parche_lung) < min_lung_percent:
                    continue
                    
                parche_ct = arr_ct_enmascarado[y:y+window_size, x:x+window_size]
                # Omitir aire puro
                if np.mean(parche_ct) < -980:
                    continue
                    
                parches_coords.append((y, x))
                parches_arrays.append(parche_ct)
                
        num_parches = len(parches_coords)
        heatmap_acc = np.zeros((alto, ancho), dtype=np.float32)
        conteo_acc = np.zeros((alto, ancho), dtype=np.float32)
        
        if num_parches > 0:
            # Extracción paralela acelerada
            features_list = joblib.Parallel(n_jobs=-1, batch_size=16)(
                joblib.delayed(extraer_caracteristicas_parche)(p, meta_features) for p in parches_arrays
            )
            
            # Predicción vectorizada con Random Forest
            X_df = pd.DataFrame(features_list, columns=top_features)
            probabilidades = rf_model.predict_proba(X_df)[:, 1]
            
            for (y, x), proba in zip(parches_coords, probabilidades):
                heatmap_acc[y:y+window_size, x:x+window_size] += proba
                conteo_acc[y:y+window_size, x:x+window_size] += 1.0
                
        # 6. Promediar superposiciones, suavizar y restringir estrictamente al pulmón
        conteo_acc[conteo_acc == 0] = 1.0
        heatmap_raw = (heatmap_acc / conteo_acc)
        
        # Aplicar leve suavizado espacial gaussiano para eliminar efecto de bloques y dejar mapa continuo
        if num_parches > 0 and stride >= 8:
            heatmap_smooth = gaussian_filter(heatmap_raw, sigma=1.0)
        else:
            heatmap_smooth = heatmap_raw
            
        heatmap_corte = heatmap_smooth * arr_lung_bin
        
        # Píxeles con P >= 0.5 dentro del pulmón
        fibrosis_pixels_corte = int(np.sum((heatmap_corte >= 0.5) & (arr_lung_bin == 1)))
        total_fibrosis_voxels += fibrosis_pixels_corte
        pct_fibrosis_corte = (fibrosis_pixels_corte / lung_pixels_corte * 100.0) if lung_pixels_corte > 0 else 0.0
        
        dt_corte = time.time() - t_slice
        print(f"    Parches: {num_parches:3d} | Fibrosis: {pct_fibrosis_corte:5.1f}% (ROI real: {roi_pixels_corte:5d} px) | Tiempo: {dt_corte:4.1f}s", flush=True)
        
        vol_ct.append(arr_ct)
        vol_lung_mask.append(arr_lung_bin)
        vol_roi_mask.append(arr_roi_bin)
        vol_heatmaps.append(heatmap_corte)
        
        slice_metrics.append({
            'slice': idx,
            'filename': item['filename'],
            'lung_pixels': lung_pixels_corte,
            'fibrosis_pixels': fibrosis_pixels_corte,
            'fibrosis_pct': pct_fibrosis_corte,
            'roi_pixels': roi_pixels_corte
        })
        
    # Convertir a matrices 3D (Z, H, W)
    vol_ct = np.array(vol_ct)
    vol_lung_mask = np.array(vol_lung_mask)
    vol_roi_mask = np.array(vol_roi_mask)
    vol_heatmaps = np.array(vol_heatmaps)
    
    # 7. Métricas globales del volumen
    pct_global_fibrosis = (total_fibrosis_voxels / total_lung_voxels * 100.0) if total_lung_voxels > 0 else 0.0
    pct_global_roi = (total_roi_voxels / total_lung_voxels * 100.0) if total_lung_voxels > 0 else 0.0
    tiempo_total_global = time.time() - t_inicio_global
    
    print("\n" + "="*60, flush=True)
    print("RESUMEN CLÍNICO VOLUMÉTRICO DE FIBROSIS (CORREGIDO)", flush=True)
    print("="*60, flush=True)
    print(f"Paciente:                          {paciente_id}", flush=True)
    print(f"Cortes Procesados:                 {num_cortes} (Tiempo total: {tiempo_total_global:.1f} s)", flush=True)
    print(f"Volumen Pulmonar Total:            {total_lung_voxels:,} voxels", flush=True)
    print(f"Volumen Fibrosis Estimada:         {total_fibrosis_voxels:,} voxels", flush=True)
    print(f"Carga Global de Fibrosis (P>=0.5): {pct_global_fibrosis:.2f}%", flush=True)
    if total_roi_voxels > 0:
        print(f"Carga Real de Fibrosis (Ground Truth): {pct_global_roi:.2f}%", flush=True)
    print("="*60 + "\n", flush=True)
    
    # 8. Guardar volumen resultante en formato .npz
    archivo_salida_npz = os.path.join(DATA_DIR, f"heatmap_volumen_paciente_{paciente_id}.npz")
    np.savez_compressed(
        archivo_salida_npz,
        vol_ct=vol_ct,
        vol_lung_mask=vol_lung_mask,
        vol_roi_mask=vol_roi_mask,
        vol_heatmaps=vol_heatmaps,
        pct_global_fibrosis=pct_global_fibrosis,
        slice_metrics=json.dumps(slice_metrics)
    )
    print(f"Volumen 3D guardado exitosamente en: {archivo_salida_npz}", flush=True)
    
    # 9. Guardar resumen gráfico estático multicorte
    guardar_resumen_estatico(vol_ct, vol_heatmaps, vol_roi_mask, slice_metrics, paciente_id, pct_global_fibrosis)
    
    return vol_ct, vol_lung_mask, vol_roi_mask, vol_heatmaps, slice_metrics, pct_global_fibrosis


def guardar_resumen_estatico(vol_ct, vol_heatmaps, vol_roi_mask, slice_metrics, paciente_id, pct_global):
    """Genera una imagen con mosaico de los cortes más representativos."""
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
        
        # Panel 1: CT
        axes[row, 0].imshow(vol_ct[idx], cmap='gray', vmin=-1000, vmax=400)
        axes[row, 0].set_title(f"Corte {m['slice']} ({m['filename']}) - TC")
        axes[row, 0].axis('off')
        
        # Panel 2: Heatmap en Pulmón
        axes[row, 1].imshow(vol_ct[idx], cmap='gray', vmin=-1000, vmax=400)
        im_h = axes[row, 1].imshow(vol_heatmaps[idx], cmap='jet', alpha=0.45, vmin=0, vmax=1)
        axes[row, 1].set_title(f"Mapa Radiomico (Fibrosis: {m['fibrosis_pct']:.1f}%)")
        axes[row, 1].axis('off')
        
        # Panel 3: Ground Truth
        axes[row, 2].imshow(vol_ct[idx], cmap='gray', vmin=-1000, vmax=400)
        if np.sum(vol_roi_mask[idx]) > 0:
            axes[row, 2].imshow(vol_roi_mask[idx], cmap='autumn', alpha=0.5)
            axes[row, 2].set_title("Ground Truth (Anotacion Medica)")
        else:
            axes[row, 2].set_title("Ground Truth (Sin fibrosis anotada)")
        axes[row, 2].axis('off')
        
    plt.suptitle(f"Analisis Volumetrico de Fibrosis - Paciente {paciente_id} (Carga Global: {pct_global:.2f}%)", fontsize=16)
    plt.tight_layout()
    
    archivo_resumen = os.path.join(DATA_DIR, f"resumen_volumen_paciente_{paciente_id}.png")
    plt.savefig(archivo_resumen, dpi=200, bbox_inches='tight')
    print(f"Resumen visual guardado en: {archivo_resumen}", flush=True)
    plt.close()


def visor_interactivo_3d(vol_ct, vol_heatmaps, vol_roi_mask, slice_metrics, paciente_id, pct_global):
    """Visor interactivo con Slider para recorrer el volumen corte por corte."""
    num_cortes = len(vol_ct)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 7))
    plt.subplots_adjust(bottom=0.20)
    
    corte_inicial = min(num_cortes - 1, max(0, num_cortes // 2))
    
    # TC
    im_ct = axes[0].imshow(vol_ct[corte_inicial], cmap='gray', vmin=-1000, vmax=400)
    axes[0].set_title(f"TC Pulmonar")
    axes[0].axis('off')
    
    # Heatmap
    axes[1].imshow(vol_ct[corte_inicial], cmap='gray', vmin=-1000, vmax=400)
    im_heat = axes[1].imshow(vol_heatmaps[corte_inicial], cmap='jet', alpha=0.45, vmin=0, vmax=1)
    title_heat = axes[1].set_title(f"Probabilidad de Fibrosis")
    axes[1].axis('off')
    
    # ROI
    axes[2].imshow(vol_ct[corte_inicial], cmap='gray', vmin=-1000, vmax=400)
    im_roi = axes[2].imshow(vol_roi_mask[corte_inicial], cmap='autumn', alpha=0.5, vmin=0, vmax=1)
    title_roi = axes[2].set_title("Ground Truth")
    axes[2].axis('off')
    
    cbar = fig.colorbar(im_heat, ax=axes[1], orientation='horizontal', fraction=0.046, pad=0.04)
    cbar.set_label("P(Fibrosis)")
    
    # Slider de corte
    ax_slider = plt.axes([0.25, 0.08, 0.50, 0.04])
    slider = Slider(ax_slider, 'Corte Axial', 0, num_cortes - 1, valinit=corte_inicial, valstep=1)
    
    def update(val):
        idx = int(slider.val)
        m = slice_metrics[idx]
        
        axes[0].imshow(vol_ct[idx], cmap='gray', vmin=-1000, vmax=400)
        axes[0].set_title(f"TC: {m['filename']} (Corte {m['slice']})")
        
        axes[1].clear()
        axes[1].imshow(vol_ct[idx], cmap='gray', vmin=-1000, vmax=400)
        axes[1].imshow(vol_heatmaps[idx], cmap='jet', alpha=0.45, vmin=0, vmax=1)
        axes[1].set_title(f"Prediccion Radiomica (Fibrosis: {m['fibrosis_pct']:.1f}%)")
        axes[1].axis('off')
        
        axes[2].clear()
        axes[2].imshow(vol_ct[idx], cmap='gray', vmin=-1000, vmax=400)
        if np.sum(vol_roi_mask[idx]) > 0:
            axes[2].imshow(vol_roi_mask[idx], cmap='autumn', alpha=0.5)
            axes[2].set_title("Ground Truth (Fibrosis Presente)")
        else:
            axes[2].set_title("Ground Truth (Sin anotacion)")
        axes[2].axis('off')
        
        fig.canvas.draw_idle()
        
    slider.on_changed(update)
    update(corte_inicial)
    
    plt.suptitle(f"Explorador Volumetrico 3D - Paciente {paciente_id} | Carga Global: {pct_global:.2f}%", fontsize=15)
    plt.show()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analisis Volumetrico 3D de Fibrosis Radiomica")
    parser.add_argument("--paciente", type=str, default=PACIENTE_DEFAULT, help="ID del paciente (ej: 77)")
    parser.add_argument("--categoria", type=str, default=CATEGORIA_DEFAULT, help="Categoria (fibrosis o healthy)")
    parser.add_argument("--stride", type=int, default=STRIDE, help="Salto en pixeles de la ventana deslizante (default: 8)")
    parser.add_argument("--min_lung", type=float, default=MIN_LUNG_PERCENT, help="Porcentaje minimo de pulmon en parche (default: 0.40)")
    parser.add_argument("--interactive", action="store_true", help="Abrir visor interactivo con slider")
    args = parser.parse_args()
    
    vol_ct, vol_lung_mask, vol_roi_mask, vol_heatmaps, slice_metrics, pct_global = procesar_volumen_paciente(
        paciente_id=args.paciente,
        categoria=args.categoria,
        stride=args.stride,
        window_size=WINDOW_SIZE,
        min_lung_percent=args.min_lung
    )
    
    if args.interactive:
        visor_interactivo_3d(vol_ct, vol_heatmaps, vol_roi_mask, slice_metrics, args.paciente, pct_global)


if __name__ == "__main__":
    main()
