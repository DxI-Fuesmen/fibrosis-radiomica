import os
import sys
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
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")

MODELO_PATH = os.path.join(MODELS_DIR, "mejor_modelo_multiclase.pkl")
TEST_JSON_PATH = os.path.join(DATA_DIR, "test_volumetrico_pacientes.json")

WINDOW_SIZE = 24
STRIDE = 12
MIN_LUNG_PERCENT = 0.35


def cargar_modelo_y_metadatos():
    if not os.path.exists(MODELO_PATH):
        raise FileNotFoundError(f"No se encontró el modelo en: {MODELO_PATH}")
    if not os.path.exists(TEST_JSON_PATH):
        raise FileNotFoundError(f"No se encontró el archivo de test en: {TEST_JSON_PATH}")
        
    artefacto = joblib.load(MODELO_PATH)
    with open(TEST_JSON_PATH, 'r', encoding='utf-8') as f:
        test_pacientes = json.load(f)
        
    return artefacto, test_pacientes


def configurar_extractor(meta_features):
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
    return extractor


def extraer_caracteristicas_parche_rapido(parche_ct, meta_features):
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
    
    sitk_img = sitk.GetImageFromArray(parche_ct)
    sitk_mask = sitk.GetImageFromArray(np.ones_like(parche_ct, dtype=np.uint8))
    try:
        res = extractor.execute(sitk_img, sitk_mask)
        return [float(res.get(k, 0.0)) for k in meta_features['features']]
    except Exception:
        return [0.0] * len(meta_features['features'])


def procesar_paciente_test(paciente_info, artefacto_modelo, stride=STRIDE):
    paciente_id = paciente_info['id']
    nii_path = paciente_info['nii_path']
    nrrd_path = paciente_info['nrrd_path']
    
    t0 = time.time()
    print(f"\n--> Evaluando {paciente_id} (Cortes: {paciente_info['slices']})...", flush=True)
    
    img_ct = sitk.ReadImage(nii_path)
    img_mask = sitk.ReadImage(nrrd_path)
    
    # Obtener espaciado físico para cálculo de volumen en cm³ (mL)
    spacing = img_ct.GetSpacing()  # (sx, sy, sz) en mm
    voxel_vol_cm3 = (spacing[0] * spacing[1] * spacing[2]) / 1000.0  # 1 cm³ = 1000 mm³
    
    arr_ct_raw = sitk.GetArrayFromImage(img_ct)
    arr_mask = sitk.GetArrayFromImage(img_mask)
    num_cortes, alto, ancho = arr_ct_raw.shape
    
    lung_mask_3d = (arr_mask > 0).astype(np.uint8)
    
    # 1. Aplicar exclusión de vasos sanguíneos intrapulmonares
    arr_ct_libre, mask_vasos, arr_hu = preparar_tc_libre_de_vasos(arr_ct_raw, lung_mask_3d)
    
    total_lung_voxels = int(np.sum(lung_mask_3d))
    vol_pulmon_total_cm3 = round(total_lung_voxels * voxel_vol_cm3, 2)
    
    real_sano_voxels = int(np.sum(arr_mask == 1))
    real_ggo_voxels = int(np.sum(arr_mask == 2))
    real_fib_voxels = int(np.sum(arr_mask == 3))
    
    vol_real_sano_cm3 = round(real_sano_voxels * voxel_vol_cm3, 2)
    vol_real_ggo_cm3 = round(real_ggo_voxels * voxel_vol_cm3, 2)
    vol_real_fib_cm3 = round(real_fib_voxels * voxel_vol_cm3, 2)
    
    pct_real_sano = round(real_sano_voxels / total_lung_voxels * 100.0, 2) if total_lung_voxels > 0 else 0
    pct_real_ggo = round(real_ggo_voxels / total_lung_voxels * 100.0, 2) if total_lung_voxels > 0 else 0
    pct_real_fib = round(real_fib_voxels / total_lung_voxels * 100.0, 2) if total_lung_voxels > 0 else 0
    
    clf = artefacto_modelo['model']
    scaler = artefacto_modelo['scaler']
    top_features = artefacto_modelo['features']
    
    step_z = 2
    cortes_evaluados = list(range(0, num_cortes, step_z))
    
    # 2. Recolectar TODOS los parches del volumen 3D en memoria
    todos_parches_coords = []  # (z, y, x)
    todos_parches_arrays = []
    
    for z in cortes_evaluados:
        slice_ct = arr_ct_libre[z]
        slice_lung = lung_mask_3d[z]
        
        if np.sum(slice_lung) < 200:
            continue
            
        for y in range(0, alto - WINDOW_SIZE + 1, stride):
            for x in range(0, ancho - WINDOW_SIZE + 1, stride):
                p_l = slice_lung[y:y+WINDOW_SIZE, x:x+WINDOW_SIZE]
                if np.mean(p_l) < MIN_LUNG_PERCENT:
                    continue
                p_c = slice_ct[y:y+WINDOW_SIZE, x:x+WINDOW_SIZE]
                if np.mean(p_c) < -980:
                    continue
                todos_parches_coords.append((z, y, x))
                todos_parches_arrays.append(p_c)
                
    num_total_parches = len(todos_parches_arrays)
    print(f"     Extrayendo radiómica en paralelo ({num_total_parches} parches 3D)...", flush=True)
    
    if num_total_parches == 0:
        return None
        
    # 3. UNA ÚNICA LLAMADA PARALELA POR PACIENTE
    feats_list = joblib.Parallel(n_jobs=-1, batch_size=32)(
        joblib.delayed(extraer_caracteristicas_parche_rapido)(p, artefacto_modelo) for p in todos_parches_arrays
    )
    
    X_df = pd.DataFrame(feats_list, columns=top_features)
    X_sc = scaler.transform(X_df.values)
    probas = clf.predict_proba(X_sc)
    
    # 4. Asignar probabilidades a cada corte z
    heatmaps_sano = {z: np.zeros((alto, ancho), dtype=np.float32) for z in cortes_evaluados}
    heatmaps_ggo = {z: np.zeros((alto, ancho), dtype=np.float32) for z in cortes_evaluados}
    heatmaps_fib = {z: np.zeros((alto, ancho), dtype=np.float32) for z in cortes_evaluados}
    conteos = {z: np.zeros((alto, ancho), dtype=np.float32) for z in cortes_evaluados}
    
    for (z, y, x), pr in zip(todos_parches_coords, probas):
        heatmaps_sano[z][y:y+WINDOW_SIZE, x:x+WINDOW_SIZE] += pr[0]
        heatmaps_ggo[z][y:y+WINDOW_SIZE, x:x+WINDOW_SIZE] += pr[1]
        heatmaps_fib[z][y:y+WINDOW_SIZE, x:x+WINDOW_SIZE] += pr[2]
        conteos[z][y:y+WINDOW_SIZE, x:x+WINDOW_SIZE] += 1.0
        
    est_sano_voxels = 0
    est_ggo_voxels = 0
    est_fib_voxels = 0
    
        # Recuperar umbrales óptimos de decisión (Youden) almacenados en el modelo
        th_sano = artefacto_modelo.get('optimal_thresholds', {}).get('sano', 0.40)
        th_ggo = artefacto_modelo.get('optimal_thresholds', {}).get('ggo', 0.35)
        th_fib = artefacto_modelo.get('optimal_thresholds', {}).get('fibrosis', 0.35)
        
        # Ponderación normalizada por umbrales de máxima sensibilidad/especificidad
        score_s = hm_s_sm / max(th_sano, 0.05)
        score_g = hm_g_sm / max(th_ggo, 0.05)
        score_f = hm_f_sm / max(th_fib, 0.05)
        
        # Asignación multiclase calibrada
        es_f = (score_f >= score_g) & (score_f >= score_s) & (hm_f_sm >= th_fib) & (slice_lung == 1)
        es_g = (score_g >= score_s) & (score_g >= score_f) & (hm_g_sm >= th_ggo) & (~es_f) & (slice_lung == 1)
        es_s = (~es_f) & (~es_g) & (slice_lung == 1)
        
        est_sano_voxels += int(np.sum(es_s))
        est_ggo_voxels += int(np.sum(es_g))
        est_fib_voxels += int(np.sum(es_f))
        
    est_total = est_sano_voxels + est_ggo_voxels + est_fib_voxels
    pct_est_sano = round(est_sano_voxels / est_total * 100.0, 2) if est_total > 0 else 0
    pct_est_ggo = round(est_ggo_voxels / est_total * 100.0, 2) if est_total > 0 else 0
    pct_est_fib = round(est_fib_voxels / est_total * 100.0, 2) if est_total > 0 else 0
    
    vol_est_sano_cm3 = round((pct_est_sano / 100.0) * vol_pulmon_total_cm3, 2)
    vol_est_ggo_cm3 = round((pct_est_ggo / 100.0) * vol_pulmon_total_cm3, 2)
    vol_est_fib_cm3 = round((pct_est_fib / 100.0) * vol_pulmon_total_cm3, 2)
    
    dt = time.time() - t0
    print(f"     Volumen Pulmonar Total: {vol_pulmon_total_cm3} cm³", flush=True)
    print(f"     Sano:     Real = {pct_real_sano:5.1f}% ({vol_real_sano_cm3:6.1f} cm³) | Estimado = {pct_est_sano:5.1f}% ({vol_est_sano_cm3:6.1f} cm³) (Diff: {pct_est_sano-pct_real_sano:+5.1f}%)", flush=True)
    print(f"     GGO:      Real = {pct_real_ggo:5.1f}% ({vol_real_ggo_cm3:6.1f} cm³) | Estimado = {pct_est_ggo:5.1f}% ({vol_est_ggo_cm3:6.1f} cm³) (Diff: {pct_est_ggo-pct_real_ggo:+5.1f}%)", flush=True)
    print(f"     Fibrosis: Real = {pct_real_fib:5.1f}% ({vol_real_fib_cm3:6.1f} cm³) | Estimado = {pct_est_fib:5.1f}% ({vol_est_fib_cm3:6.1f} cm³) (Diff: {pct_est_fib-pct_real_fib:+5.1f}%)", flush=True)
    print(f"     Tiempo:   {dt:.1f} s", flush=True)
    
    return {
        'id': paciente_id,
        'slices': num_cortes,
        'volumen_pulmonar_cm3': vol_pulmon_total_cm3,
        'sano_real_pct': pct_real_sano,
        'sano_est_pct': pct_est_sano,
        'vol_sano_real_cm3': vol_real_sano_cm3,
        'vol_sano_est_cm3': vol_est_sano_cm3,
        'error_sano_pct': round(abs(pct_est_sano - pct_real_sano), 2),
        'ggo_real_pct': pct_real_ggo,
        'ggo_est_pct': pct_est_ggo,
        'vol_ggo_real_cm3': vol_real_ggo_cm3,
        'vol_ggo_est_cm3': vol_est_ggo_cm3,
        'error_ggo_pct': round(abs(pct_est_ggo - pct_real_ggo), 2),
        'fib_real_pct': pct_real_fib,
        'fib_est_pct': pct_est_fib,
        'vol_fib_real_cm3': vol_real_fib_cm3,
        'vol_fib_est_cm3': vol_est_fib_cm3,
        'error_fib_pct': round(abs(pct_est_fib - pct_real_fib), 2),
        'tiempo_s': round(dt, 1)
    }


def generar_reporte_patrones_parenquimatosos(df_res):
    """
    Exporta reporte detallado de patrones parenquimatosos por paciente en CSV y JSON.
    """
    csv_path = os.path.join(DATA_DIR, "reporte_volumetrico_pacientes.csv")
    json_path = os.path.join(DATA_DIR, "reporte_volumetrico_pacientes.json")
    
    df_res.to_csv(csv_path, index=False)
    
    res_dict = {
        'total_pacientes': len(df_res),
        'volumen_promedio_pulmon_cm3': round(float(df_res['volumen_pulmonar_cm3'].mean()), 2),
        'mae_sano_pct': round(float(df_res['error_sano_pct'].mean()), 2),
        'mae_ggo_pct': round(float(df_res['error_ggo_pct'].mean()), 2),
        'mae_fib_pct': round(float(df_res['error_fib_pct'].mean()), 2),
        'correlacion_fibrosis_r': round(float(np.corrcoef(df_res['fib_real_pct'], df_res['fib_est_pct'])[0, 1]), 4),
        'pacientes': df_res.to_dict(orient='records')
    }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(res_dict, f, indent=2)
        
    print(f"\n-> Reporte de patrones parenquimatosos guardado en:")
    print(f"   CSV:  {csv_path}")
    print(f"   JSON: {json_path}", flush=True)


def graficar_distribucion_patrones_apilados(df_res):
    """
    Genera un gráfico de barras apiladas comparando la distribución Real vs Estimada
    de patrones parenquimatosos (Sano, GGO, Fibrosis) por paciente (estilo Yang et al. / Zhao et al.).
    """
    n_pacientes = len(df_res)
    indices = np.arange(n_pacientes)
    width = 0.38
    
    fig, ax = plt.subplots(figsize=(15, 6))
    
    colores = {
        'Sano': '#2b5c8f',       # Azul oscuro
        'GGO': '#f39c12',        # Naranja / Amarillo
        'Fibrosis': '#c0392b'    # Rojo
    }
    
    # Barras Reales (Ground Truth)
    p1 = ax.bar(indices - width/2, df_res['sano_real_pct'], width, label='Real: Sano', color=colores['Sano'], alpha=0.9, edgecolor='black')
    p2 = ax.bar(indices - width/2, df_res['ggo_real_pct'], width, bottom=df_res['sano_real_pct'], label='Real: GGO', color=colores['GGO'], alpha=0.9, edgecolor='black')
    bottom_fib_real = df_res['sano_real_pct'] + df_res['ggo_real_pct']
    p3 = ax.bar(indices - width/2, df_res['fib_real_pct'], width, bottom=bottom_fib_real, label='Real: Fibrosis', color=colores['Fibrosis'], alpha=0.9, edgecolor='black')
    
    # Barras Estimadas (Radiómica)
    p4 = ax.bar(indices + width/2, df_res['sano_est_pct'], width, label='Est: Sano', color=colores['Sano'], alpha=0.45, hatch='//', edgecolor='black')
    p5 = ax.bar(indices + width/2, df_res['ggo_est_pct'], width, bottom=df_res['sano_est_pct'], label='Est: GGO', color=colores['GGO'], alpha=0.45, hatch='//', edgecolor='black')
    bottom_fib_est = df_res['sano_est_pct'] + df_res['ggo_est_pct']
    p6 = ax.bar(indices + width/2, df_res['fib_est_pct'], width, bottom=bottom_fib_est, label='Est: Fibrosis', color=colores['Fibrosis'], alpha=0.45, hatch='//', edgecolor='black')
    
    ax.set_ylabel('Proporción Volumétrica Pulmonar (%)', fontsize=12, fontweight='bold')
    ax.set_title('Cuantificación Global de Patrones Parenquimatosos en Test Ciego (Real vs Estimado)', fontsize=14, fontweight='bold')
    ax.set_xticks(indices)
    ax.set_xticklabels([f"{row['id']}\n({row['volumen_pulmonar_cm3']} cm³)" for _, row in df_res.iterrows()], fontsize=10)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    
    # Leyenda personalizada
    handles = [
        plt.Rectangle((0,0),1,1, color=colores['Sano'], alpha=0.9, ec='k'),
        plt.Rectangle((0,0),1,1, color=colores['GGO'], alpha=0.9, ec='k'),
        plt.Rectangle((0,0),1,1, color=colores['Fibrosis'], alpha=0.9, ec='k'),
        plt.Rectangle((0,0),1,1, color='gray', alpha=0.9, ec='k'),
        plt.Rectangle((0,0),1,1, color='gray', alpha=0.45, hatch='//', ec='k')
    ]
    labels = ['Sano', 'Vidrio Esmerilado (GGO)', 'Fibrosis Pulmonar', 'Ground Truth (Sólido)', 'Predicción Radiómica (Rayado)']
    ax.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.0, 1.18), ncol=5, frameon=True, fontsize=10)
    
    plt.tight_layout()
    out_img = os.path.join(DATA_DIR, "distribucion_patrones_parenquimatosos_test.png")
    plt.savefig(out_img, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"-> Gráfico de patrones apilados guardado en: {out_img}", flush=True)


def graficar_resultados_test(df_res):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colores = {'Sano': '#2b5c8f', 'GGO': '#d4ac0d', 'Fibrosis': '#c0392b'}
    
    # 1. Sano
    axes[0].scatter(df_res['sano_real_pct'], df_res['sano_est_pct'], color=colores['Sano'], s=80, edgecolors='black')
    axes[0].plot([0, 100], [0, 100], 'k--', alpha=0.5, label='Ideal (y=x)')
    axes[0].set_title(f"Parénquima Sano\nMAE: {df_res['error_sano_pct'].mean():.1f}%", fontsize=12)
    axes[0].set_xlabel("Ground Truth Real (%)")
    axes[0].set_ylabel("Estimación Radiómica (%)")
    axes[0].set_xlim(0, 100)
    axes[0].set_ylim(0, 100)
    axes[0].grid(True, linestyle=':', alpha=0.6)
    
    # 2. GGO
    axes[1].scatter(df_res['ggo_real_pct'], df_res['ggo_est_pct'], color=colores['GGO'], s=80, edgecolors='black')
    axes[1].plot([0, 50], [0, 50], 'k--', alpha=0.5, label='Ideal (y=x)')
    axes[1].set_title(f"Vidrio Esmerilado (GGO)\nMAE: {df_res['error_ggo_pct'].mean():.1f}%", fontsize=12)
    axes[1].set_xlabel("Ground Truth Real (%)")
    axes[1].set_ylabel("Estimación Radiómica (%)")
    axes[1].set_xlim(0, 50)
    axes[1].set_ylim(0, 50)
    axes[1].grid(True, linestyle=':', alpha=0.6)
    
    # 3. Fibrosis
    r_corr = np.corrcoef(df_res['fib_real_pct'], df_res['fib_est_pct'])[0, 1] if len(df_res) > 1 else 0
    axes[2].scatter(df_res['fib_real_pct'], df_res['fib_est_pct'], color=colores['Fibrosis'], s=80, edgecolors='black')
    axes[2].plot([0, 50], [0, 50], 'k--', alpha=0.5, label='Ideal (y=x)')
    axes[2].set_title(f"Fibrosis Establecida\nMAE: {df_res['error_fib_pct'].mean():.1f}% | r = {r_corr:.2f}", fontsize=12)
    axes[2].set_xlabel("Ground Truth Real (%)")
    axes[2].set_ylabel("Estimación Radiómica (%)")
    axes[2].set_xlim(0, 50)
    axes[2].set_ylim(0, 50)
    axes[2].grid(True, linestyle=':', alpha=0.6)
    
    plt.suptitle("Validación Volumétrica 3D Independiente (Test Ciego - 8 Pacientes)", fontsize=14, y=1.03)
    plt.tight_layout()
    
    out_img = os.path.join(DATA_DIR, "validacion_volumetrica_test.png")
    plt.savefig(out_img, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"-> Gráfico de validación volumétrica guardado en: {out_img}", flush=True)


def main():
    print("="*80, flush=True)
    print("VALIDACIÓN VOLUMÉTRICA 3D CON EXCLUSIÓN VASCULAR (TEST CIEGO)", flush=True)
    print("="*80, flush=True)
    
    artefacto_modelo, test_pacientes = cargar_modelo_y_metadatos()
    
    resultados = []
    for p_info in test_pacientes:
        res = procesar_paciente_test(p_info, artefacto_modelo)
        if res:
            resultados.append(res)
            
    df_res = pd.DataFrame(resultados)
    
    print("\n" + "="*95, flush=True)
    print("RESUMEN DE PATRONES PARENQUIMATOSOS Y VOLUMETRÍA CLÍNICA (Test Set Independiente)", flush=True)
    print("="*95, flush=True)
    print(f"{'Paciente':<9} {'Vol_Total_cm³':<14} {'Sano_Real_%':<12} {'Sano_Est_%':<12} {'GGO_Real_%':<11} {'GGO_Est_%':<11} {'Fib_Real_%':<11} {'Fib_Est_%':<11} {'Err_Fib_%':<10}")
    for _, row in df_res.iterrows():
        print(f"{row['id']:<9} {row['volumen_pulmonar_cm3']:<14.1f} {row['sano_real_pct']:<12.1f} {row['sano_est_pct']:<12.1f} {row['ggo_real_pct']:<11.1f} {row['ggo_est_pct']:<11.1f} {row['fib_real_pct']:<11.1f} {row['fib_est_pct']:<11.1f} {row['error_fib_pct']:<10.1f}")
        
    mae_sano = df_res['error_sano_pct'].mean()
    mae_ggo = df_res['error_ggo_pct'].mean()
    mae_fib = df_res['error_fib_pct'].mean()
    corr_fib = np.corrcoef(df_res['fib_real_pct'], df_res['fib_est_pct'])[0, 1] if len(df_res) > 1 else 0
    
    print("="*95, flush=True)
    print(f"Error Absoluto Medio en Fibrosis (MAE Fib): {mae_fib:.2f}% | Correlación R = {corr_fib:.4f} (R² = {corr_fib**2:.4f})")
    print(f"Error Absoluto Medio en GGO (MAE GGO):      {mae_ggo:.2f}%")
    print(f"Error Absoluto Medio en Sano (MAE Sano):    {mae_sano:.2f}%")
    print("="*95, flush=True)
    
    # 1. Gráficos de dispersión y correlación
    graficar_resultados_test(df_res)
    # 2. Gráfico de barras apiladas de distribución parenquimatosa (Yang et al. / Zhao et al.)
    graficar_distribucion_patrones_apilados(df_res)
    # 3. Exportar reportes estructurados CSV y JSON
    generar_reporte_patrones_parenquimatosos(df_res)


if __name__ == "__main__":
    main()
