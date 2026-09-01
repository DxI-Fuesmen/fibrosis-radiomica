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
# 1. CONFIGURACIÓN Y RUTAS
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

BASE_REPO = os.path.join(BASE_DIR, "..", "base")
DIR_SSC = os.path.join(BASE_REPO, "SSc-ILD")
DIR_ILD = os.path.join(BASE_REPO, "ILD_DB", "ILD_DB_Clasificada")

WINDOW_SIZE = 24  # Alta resolución espacial (16 mm x 16 mm)
PATCHES_PER_CLASS_TARGET = 1200

# 8 Pacientes de SSc-ILD reservados exclusivamente para Test Volumétrico Ciego
TEST_PATIENT_INDICES = [3, 8, 14, 20, 25, 32, 37, 40]


def configurar_extractor_completo():
    """Configura PyRadiomics para extraer características Originales, Wavelets, LoG, Gradient y Square."""
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.disableAllImageTypes()
    extractor.disableAllFeatures()
    
    extractor.enableImageTypeByName('Original')
    extractor.enableImageTypeByName('Wavelet')
    extractor.enableImageTypeByName('Square')
    extractor.enableImageTypeByName('Gradient')
    extractor.enableImageTypeByName('LoG', customArgs={'sigma': [1.0, 2.0, 3.0, 5.0]})
    
    for feat_cls in ['firstorder', 'glcm', 'glrlm', 'glszm', 'gldm', 'ngtdm']:
        extractor.enableFeatureClassByName(feat_cls)
        
    extractor.settings['binWidth'] = 25.0
    extractor.settings['minimumROIDimensions'] = 2
    return extractor


def extraer_caracteristicas_un_parche(parche_ct, extractor):
    """Extrae todas las características radiómicas para un parche 2D."""
    sitk_img = sitk.GetImageFromArray(parche_ct)
    sitk_mask = sitk.GetImageFromArray(np.ones_like(parche_ct, dtype=np.uint8))
    try:
        resultado = extractor.execute(sitk_img, sitk_mask)
        feats = {k: float(v) for k, v in resultado.items() if not k.startswith('diagnostics_')}
        return feats
    except Exception:
        return None


def recolectar_parches_ssc_ild(test_indices=TEST_PATIENT_INDICES):
    """Recolecta parches libres de vasos de los 32 pacientes de SSc-ILD asignados a entrenamiento."""
    print("="*60, flush=True)
    print("RECOLECTANDO PARCHES LIBRES DE VASOS (SSc-ILD)", flush=True)
    print(f"Resolución de Ventana: {WINDOW_SIZE}x{WINDOW_SIZE} px", flush=True)
    print("="*60, flush=True)
    
    parches_sano = []
    parches_ggo = []
    parches_fib = []
    test_patient_info = []
    
    for i in range(1, 41):
        nii_path = os.path.join(DIR_SSC, f"{i}.nii")
        nrrd_path = os.path.join(DIR_SSC, f"{i}.nrrd")
        
        if not os.path.exists(nii_path) or not os.path.exists(nrrd_path):
            continue
            
        paciente_id = f"SSc_{i:02d}"
        img_ct = sitk.ReadImage(nii_path)
        img_mask = sitk.ReadImage(nrrd_path)
        
        arr_ct_raw = sitk.GetArrayFromImage(img_ct)
        arr_mask = sitk.GetArrayFromImage(img_mask)
        num_cortes, alto, ancho = arr_ct_raw.shape
        
        # Si es paciente de test, registrar estadísticas y NO extraer parches
        if i in test_indices:
            total_lung = int(np.sum(arr_mask > 0))
            sano = int(np.sum(arr_mask == 1))
            ggo = int(np.sum(arr_mask == 2))
            fib = int(np.sum(arr_mask == 3))
            test_patient_info.append({
                'id': paciente_id,
                'num': i,
                'nii_path': nii_path,
                'nrrd_path': nrrd_path,
                'slices': num_cortes,
                'total_lung_voxels': total_lung,
                'sano_pct': round(sano / total_lung * 100, 2) if total_lung > 0 else 0,
                'ggo_pct': round(ggo / total_lung * 100, 2) if total_lung > 0 else 0,
                'fib_pct': round(fib / total_lung * 100, 2) if total_lung > 0 else 0,
            })
            continue
            
        # Preparar TC libre de vasos para los 32 pacientes de entrenamiento
        lung_mask = (arr_mask > 0).astype(np.uint8)
        arr_ct_libre, mask_vasos, arr_hu = preparar_tc_libre_de_vasos(arr_ct_raw, lung_mask)
        
        # Extraer parches de parénquima puro
        paso_z = 2
        for z in range(0, num_cortes, paso_z):
            slice_ct = arr_ct_libre[z]
            slice_mask = arr_mask[z]
            slice_vasos = mask_vasos[z]
            
            if np.sum(slice_mask > 0) < 400:
                continue
                
            stride = 12
            for y in range(0, alto - WINDOW_SIZE + 1, stride):
                for x in range(0, ancho - WINDOW_SIZE + 1, stride):
                    p_mask = slice_mask[y:y+WINDOW_SIZE, x:x+WINDOW_SIZE]
                    p_ct = slice_ct[y:y+WINDOW_SIZE, x:x+WINDOW_SIZE]
                    p_vasos = slice_vasos[y:y+WINDOW_SIZE, x:x+WINDOW_SIZE]
                    
                    if np.mean(p_ct) < -980 or np.mean(p_ct) > 100:
                        continue
                        
                    total_px = WINDOW_SIZE * WINDOW_SIZE
                    pct_sano = np.sum(p_mask == 1) / total_px
                    pct_ggo = np.sum(p_mask == 2) / total_px
                    pct_fib = np.sum(p_mask == 3) / total_px
                    pct_vasos = np.sum(p_vasos == 1) / total_px
                    
                    # Descartar parches dominados por vasos
                    if pct_vasos > 0.40:
                        continue
                        
                    # 1. Sano
                    if pct_sano >= 0.85 and -920 <= np.mean(p_ct) <= -650:
                        parches_sano.append({
                            'id_paciente': paciente_id,
                            'origen': 'SSc-ILD',
                            'corte_z': z,
                            'clase': 1,
                            'clase_nombre': 'Sano',
                            'array': p_ct
                        })
                    # 2. GGO
                    elif pct_ggo >= 0.50 and -800 <= np.mean(p_ct) <= -350:
                        parches_ggo.append({
                            'id_paciente': paciente_id,
                            'origen': 'SSc-ILD',
                            'corte_z': z,
                            'clase': 2,
                            'clase_nombre': 'GGO',
                            'array': p_ct
                        })
                    # 3. Fibrosis
                    elif pct_fib >= 0.50 and -750 <= np.mean(p_ct) <= -200:
                        parches_fib.append({
                            'id_paciente': paciente_id,
                            'origen': 'SSc-ILD',
                            'corte_z': z,
                            'clase': 3,
                            'clase_nombre': 'Fibrosis',
                            'array': p_ct
                        })
                        
    archivo_test_json = os.path.join(DATA_DIR, "test_volumetrico_pacientes.json")
    with open(archivo_test_json, 'w', encoding='utf-8') as f:
        json.dump(test_patient_info, f, indent=2)
    print(f"-> 8 Pacientes de Test Volumétrico guardados en: {archivo_test_json}", flush=True)
    print(f"Parches extraídos de SSc-ILD -> Sano: {len(parches_sano)}, GGO: {len(parches_ggo)}, Fibrosis: {len(parches_fib)}", flush=True)
    return parches_sano, parches_ggo, parches_fib


def recolectar_parches_ild_clasificada():
    """Recolecta parches de apoyo libres de vasos de ILD_DB_Clasificada."""
    print("\n" + "="*60, flush=True)
    print("RECOLECTANDO PARCHES DE APOYO LIBRES DE VASOS (ILD_DB_Clasificada)", flush=True)
    print("="*60, flush=True)
    
    parches_sano = []
    parches_ggo = []
    parches_fib = []
    
    # 1. Sanos
    dir_healthy = os.path.join(DIR_ILD, "healthy")
    if os.path.exists(dir_healthy):
        for pac in os.listdir(dir_healthy):
            pac_dir = os.path.join(dir_healthy, pac)
            dcms = glob.glob(os.path.join(pac_dir, "*.dcm"))
            for d in dcms:
                try:
                    arr = to_hu(sitk.GetArrayFromImage(sitk.ReadImage(d))[0])
                    lung_mask = (arr > -950) & (arr < -500)
                    arr_libre, mask_v, _ = preparar_tc_libre_de_vasos(arr, lung_mask)
                    h, w = arr_libre.shape
                    for y in range(h//4, 3*h//4, 16):
                        for x in range(w//4, 3*w//4, 16):
                            p = arr_libre[y:y+WINDOW_SIZE, x:x+WINDOW_SIZE]
                            if p.shape == (WINDOW_SIZE, WINDOW_SIZE) and -900 <= np.mean(p) <= -680 and np.std(p) > 15:
                                parches_sano.append({
                                    'id_paciente': f'ILD_H_{pac}',
                                    'origen': 'ILD_DB',
                                    'corte_z': os.path.basename(d),
                                    'clase': 1,
                                    'clase_nombre': 'Sano',
                                    'array': p
                                })
                except Exception:
                    continue
                    
    # 2. Ground Glass
    dir_gg = os.path.join(DIR_ILD, "ground_glass")
    if os.path.exists(dir_gg):
        for pac in os.listdir(dir_gg)[:15]:
            pac_dir = os.path.join(dir_gg, pac)
            dir_roi = os.path.join(pac_dir, "roi_mask")
            dcms = [f for f in glob.glob(os.path.join(pac_dir, "*.dcm")) if "mask" not in f.lower()]
            for d in dcms:
                idx_match = re.search(r'(\d+)\.dcm$', d)
                if not idx_match:
                    continue
                idx = idx_match.group(1)
                roi_files = glob.glob(os.path.join(dir_roi, f"roi_mask_*_{int(idx)}.dcm"))
                if not roi_files:
                    continue
                try:
                    arr_ct = to_hu(sitk.GetArrayFromImage(sitk.ReadImage(d))[0])
                    arr_roi = sitk.GetArrayFromImage(sitk.ReadImage(roi_files[0]))[0]
                    lung_mask = (arr_ct > -980) & (arr_ct < 100)
                    arr_libre, mask_v, _ = preparar_tc_libre_de_vasos(arr_ct, lung_mask)
                    h, w = arr_libre.shape
                    for y in range(0, h-WINDOW_SIZE+1, 12):
                        for x in range(0, w-WINDOW_SIZE+1, 12):
                            p_roi = arr_roi[y:y+WINDOW_SIZE, x:x+WINDOW_SIZE]
                            p_ct = arr_libre[y:y+WINDOW_SIZE, x:x+WINDOW_SIZE]
                            if np.mean(p_roi > 0) >= 0.50 and -800 <= np.mean(p_ct) <= -350:
                                parches_ggo.append({
                                    'id_paciente': f'ILD_GG_{pac}',
                                    'origen': 'ILD_DB',
                                    'corte_z': os.path.basename(d),
                                    'clase': 2,
                                    'clase_nombre': 'GGO',
                                    'array': p_ct
                                })
                except Exception:
                    continue

    # 3. Fibrosis
    dir_fib = os.path.join(DIR_ILD, "fibrosis")
    if os.path.exists(dir_fib):
        for pac in os.listdir(dir_fib)[:15]:
            pac_dir = os.path.join(dir_fib, pac)
            dir_roi = os.path.join(pac_dir, "roi_mask")
            dcms = [f for f in glob.glob(os.path.join(pac_dir, "*.dcm")) if "mask" not in f.lower()]
            for d in dcms:
                idx_match = re.search(r'(\d+)\.dcm$', d)
                if not idx_match:
                    continue
                idx = idx_match.group(1)
                roi_files = glob.glob(os.path.join(dir_roi, f"roi_mask_*_{int(idx)}.dcm"))
                if not roi_files:
                    continue
                try:
                    arr_ct = to_hu(sitk.GetArrayFromImage(sitk.ReadImage(d))[0])
                    arr_roi = sitk.GetArrayFromImage(sitk.ReadImage(roi_files[0]))[0]
                    lung_mask = (arr_ct > -980) & (arr_ct < 100)
                    arr_libre, mask_v, _ = preparar_tc_libre_de_vasos(arr_ct, lung_mask)
                    h, w = arr_libre.shape
                    for y in range(0, h-WINDOW_SIZE+1, 12):
                        for x in range(0, w-WINDOW_SIZE+1, 12):
                            p_roi = arr_roi[y:y+WINDOW_SIZE, x:x+WINDOW_SIZE]
                            p_ct = arr_libre[y:y+WINDOW_SIZE, x:x+WINDOW_SIZE]
                            if np.mean(p_roi > 0) >= 0.50 and -750 <= np.mean(p_ct) <= -200:
                                parches_fib.append({
                                    'id_paciente': f'ILD_Fib_{pac}',
                                    'origen': 'ILD_DB',
                                    'corte_z': os.path.basename(d),
                                    'clase': 3,
                                    'clase_nombre': 'Fibrosis',
                                    'array': p_ct
                                })
                except Exception:
                    continue

    print(f"Parches de apoyo ILD_DB -> Sano: {len(parches_sano)}, GGO: {len(parches_ggo)}, Fibrosis: {len(parches_fib)}", flush=True)
    return parches_sano, parches_ggo, parches_fib


def balancear_y_unificar(sano_ssc, ggo_ssc, fib_ssc, sano_ild, ggo_ild, fib_ild, target_n=PATCHES_PER_CLASS_TARGET):
    np.random.seed(42)
    todos_sano = sano_ssc + sano_ild
    todos_ggo = ggo_ssc + ggo_ild
    todos_fib = fib_ssc + fib_ild
    
    n_min = min(len(todos_sano), len(todos_ggo), len(todos_fib), target_n)
    print(f"\nBalanceando dataset a {n_min} parches por clase (Total: {n_min*3} parches)...", flush=True)
    
    sel_sano = list(np.random.choice(todos_sano, size=n_min, replace=False))
    sel_ggo = list(np.random.choice(todos_ggo, size=n_min, replace=False))
    sel_fib = list(np.random.choice(todos_fib, size=n_min, replace=False))
    
    dataset_parches = sel_sano + sel_ggo + sel_fib
    np.random.shuffle(dataset_parches)
    return dataset_parches


def main():
    t_inicio = time.time()
    sano_ssc, ggo_ssc, fib_ssc = recolectar_parches_ssc_ild()
    sano_ild, ggo_ild, fib_ild = recolectar_parches_ild_clasificada()
    
    dataset_parches = balancear_y_unificar(sano_ssc, ggo_ssc, fib_ssc, sano_ild, ggo_ild, fib_ild)
    
    print("\n" + "="*60, flush=True)
    print("EXTRAYENDO CARACTERÍSTICAS RADIÓMICAS MULTICLASE AVANZADAS", flush=True)
    print("="*60, flush=True)
    
    extractor = configurar_extractor_completo()
    
    def worker(item):
        feats = extraer_caracteristicas_un_parche(item['array'], extractor)
        if feats is None:
            return None
        meta = {
            'ID_Paciente': item['id_paciente'],
            'Origen': item['origen'],
            'Corte_Z': item['corte_z'],
            'Clase': item['clase'],
            'Clase_Nombre': item['clase_nombre']
        }
        meta.update(feats)
        return meta

    print(f"Extrayendo características para {len(dataset_parches)} parches en paralelo...", flush=True)
    resultados = joblib.Parallel(n_jobs=-1, batch_size=16)(
        joblib.delayed(worker)(item) for item in dataset_parches
    )
    
    resultados_validos = [r for r in resultados if r is not None]
    df = pd.DataFrame(resultados_validos)
    
    salida_csv = os.path.join(DATA_DIR, "dataset_maestro_radiomica.csv")
    df.to_csv(salida_csv, index=False)
    
    dt = time.time() - t_inicio
    print("\n" + "="*60, flush=True)
    print("EXTRACCIÓN RADIÓMICA LIBRE DE VASOS COMPLETADA", flush=True)
    print("="*60, flush=True)
    print(f"Parches válidos procesados: {len(df)}", flush=True)
    print(f"Características extraídas por parche: {len(df.columns) - 5}", flush=True)
    print(f"Distribución de Clases:\n{df['Clase_Nombre'].value_counts()}", flush=True)
    print(f"Pacientes únicos en entrenamiento: {df['ID_Paciente'].nunique()}", flush=True)
    print(f"Archivo guardado en: {salida_csv}", flush=True)
    print(f"Tiempo total: {dt:.1f} s", flush=True)
    print("="*60 + "\n", flush=True)


if __name__ == "__main__":
    main()
