import os
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import (
    binary_dilation, distance_transform_edt, 
    generate_binary_structure, label
)

def to_hu(arr):
    """Normaliza cualquier array tomográfico a Unidades Hounsfield reales [-1024, +3071]."""
    if np.min(arr) >= -50 and np.mean(arr) > 0:
        return (arr - 1024.0).astype(np.float32)
    return arr.astype(np.float32)


def segmentar_vasos_intrapulmonares(arr_ct_hu, mask_pulmon, umbral_hu=-350.0, proteger_subpleural_px=6):
    """
    Segmenta automáticamente el árbol vascular intrapulmonar de forma vectorizada y ultra-rápida.
    
    Parámetros:
    - arr_ct_hu: Array 2D o 3D con intensidades en Unidades Hounsfield.
    - mask_pulmon: Máscara binaria del pulmón (1 = pulmón, 0 = exterior).
    - umbral_hu: Umbral de corte de densidad vascular (default: -350 HU).
    - proteger_subpleural_px: Distancia en píxeles desde la pleura hacia adentro donde
      la fibrosis subpleural es protegida.
      
    Retorna:
    - mask_vasos: Máscara binaria (1 = vaso sanguíneo, 0 = parénquima).
    """
    pulmon_bin = (mask_pulmon > 0).astype(np.uint8)
    if np.sum(pulmon_bin) == 0:
        return np.zeros_like(pulmon_bin, dtype=np.uint8)
        
    is_2d = (arr_ct_hu.ndim == 2)
    if is_2d:
        arr_ct_hu = np.expand_dims(arr_ct_hu, 0)
        pulmon_bin = np.expand_dims(pulmon_bin, 0)
        
    num_cortes, alto, ancho = arr_ct_hu.shape
    mask_vasos = np.zeros((num_cortes, alto, ancho), dtype=np.uint8)
    struct_2d = generate_binary_structure(2, 1)
    
    for z in range(num_cortes):
        corte_ct = arr_ct_hu[z]
        corte_lung = pulmon_bin[z]
        
        if np.sum(corte_lung) < 100:
            continue
            
        # 1. Candidatos vasculares por densidad intrapulmonar
        candidatos = (corte_ct >= umbral_hu) & (corte_ct <= 250.0) & (corte_lung == 1)
        if not np.any(candidatos):
            continue
            
        # 2. Mapa de distancia a la pleura
        dist_pleura = distance_transform_edt(corte_lung)
        
        # 3. Vasos intrapulmonares centrales/medios (alejados de la pleura)
        vasos_directos = candidatos & (dist_pleura >= proteger_subpleural_px)
        
        # 4. Vasos grandes y ramificados conectados al hilio/centro (> 12 px de pleura)
        labeled, num_objs = label(candidatos, structure=struct_2d)
        if num_objs > 0:
            seeds = np.unique(labeled[candidatos & (dist_pleura >= 12)])
            seeds = seeds[seeds > 0]
            if len(seeds) > 0:
                vasos_conectados = np.isin(labeled, seeds)
            else:
                vasos_conectados = np.zeros_like(candidatos)
        else:
            vasos_conectados = np.zeros_like(candidatos)
            
        vasos_total = vasos_directos | vasos_conectados
        
        # 5. Dilatación de 1 píxel para eliminar el gradiente perivascular que causaba falso GGO
        mask_vasos[z] = binary_dilation(vasos_total, structure=struct_2d, iterations=1) & (corte_lung == 1)
        
    if is_2d:
        mask_vasos = mask_vasos[0]
        
    return mask_vasos.astype(np.uint8)


def inpaint_vasos_parenquima(arr_ct_hu, mask_pulmon, mask_vasos, baseline_hu=-850.0):
    """
    Reemplaza los vóxeles de vasos sanguíneos con la densidad basal de parénquima sano (-850 HU).
    """
    arr_inpainted = arr_ct_hu.copy()
    arr_inpainted[mask_pulmon == 0] = -1000.0
    arr_inpainted[(mask_pulmon == 1) & (mask_vasos == 1)] = baseline_hu
    return arr_inpainted.astype(np.float32)


def preparar_tc_libre_de_vasos(arr_ct, mask_pulmon, umbral_hu=-350.0):
    arr_hu = to_hu(arr_ct)
    pulmon_bin = (mask_pulmon > 0).astype(np.uint8)
    mask_vasos = segmentar_vasos_intrapulmonares(arr_hu, pulmon_bin, umbral_hu=umbral_hu)
    arr_libre = inpaint_vasos_parenquima(arr_hu, pulmon_bin, mask_vasos)
    return arr_libre, mask_vasos, arr_hu
