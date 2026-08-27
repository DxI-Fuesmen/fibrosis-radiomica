import os
import re
import glob
import shutil
import argparse
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from scipy.ndimage import label
from lungmask import LMInferer
import pydicom
from pydicom.uid import generate_uid

# ==========================================
# 1. CONFIGURACIÓN Y RUTAS
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

RUTA_DEFAULT_TC = r"C:\Users\malcalde\Documents\Base Pulmon\ILD_DB\ILD_DB_Clasificada\fibrosis\77"


def extraer_numero_corte(nombre_archivo):
    match = re.search(r'(\d+)\.dcm$', nombre_archivo, re.IGNORECASE)
    return int(match.group(1)) if match else 0


def leer_serie_dicom(ruta_entrada):
    """
    Lee una tomografía desde un archivo individual o una carpeta de DICOMs.
    Retorna la imagen SimpleITK 3D, la lista ordenada de archivos DICOM y el lector.
    """
    if os.path.isfile(ruta_entrada):
        print(f"Cargando archivo único: {ruta_entrada}...", flush=True)
        img_sitk = sitk.ReadImage(ruta_entrada)
        archivos_dcm = [ruta_entrada] if ruta_entrada.lower().endswith('.dcm') else []
        return img_sitk, archivos_dcm

    elif os.path.isdir(ruta_entrada):
        print(f"Buscando cortes DICOM en: {ruta_entrada}...", flush=True)
        # Buscar directamente todos los archivos .dcm
        archivos_dcm = glob.glob(os.path.join(ruta_entrada, "*.dcm"))
        if not archivos_dcm:
            # Buscar recursivamente si están en subcarpetas
            archivos_dcm = glob.glob(os.path.join(ruta_entrada, "**", "*.dcm"), recursive=True)
            # Filtrar máscaras si ya existen
            archivos_dcm = [f for f in archivos_dcm if "mask" not in os.path.basename(f).lower()]
            
        if not archivos_dcm:
            raise FileNotFoundError(f"No se encontraron archivos DICOM (.dcm) en: {ruta_entrada}")
            
        # Ordenar por índice de corte
        archivos_dcm.sort(key=extraer_numero_corte)
        print(f"Encontrados {len(archivos_dcm)} cortes DICOM ordenados.", flush=True)
        
        # Leer serie 3D con ImageSeriesReader para preservar geometría espacial exacta
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(archivos_dcm)
        img_sitk = reader.Execute()
        
        return img_sitk, archivos_dcm
    else:
        raise FileNotFoundError(f"La ruta especificada no existe: {ruta_entrada}")


def segmentar_pulmones_y_vias(img_sitk):
    """
    Aplica el modelo U-Net de lungmask para segmentar los pulmones (excluyendo tráquea)
    y aísla automáticamente la máscara de las vías aéreas centrales (tráquea y bronquios).
    """
    print("\nIniciando segmentación con Deep Learning (lungmask U-Net)...", flush=True)
    inferer = LMInferer()
    seg_array = inferer.apply(img_sitk)
    
    if seg_array.ndim == 2:
        seg_array = np.expand_dims(seg_array, 0)
        
    arr_ct = sitk.GetArrayFromImage(img_sitk)
    if arr_ct.ndim == 2:
        arr_ct = np.expand_dims(arr_ct, 0)
        
    # 1. Máscara binaria del parénquima pulmonar (valores 0 y 255)
    lung_mask = (seg_array > 0).astype(np.uint8) * 255
    
    # 2. Extracción de Vías Aéreas Centrales (Tráquea y Bronquios principales)
    print("Extrayendo máscara anatómica de vías aéreas centrales...", flush=True)
    airway_mask = np.zeros_like(lung_mask, dtype=np.uint8)
    
    num_cortes, alto, ancho = arr_ct.shape
    
    for z in range(num_cortes):
        corte_ct = arr_ct[z]
        corte_lung = (lung_mask[z] > 0)
        
        # Umbral de aire (-1050 a -920 HU) fuera de los pulmones
        aire_mediastino = (corte_ct < -920) & (corte_ct > -1050) & (~corte_lung)
        
        labeled, num_features = label(aire_mediastino)
        for i in range(1, num_features + 1):
            comp = (labeled == i)
            comp_size = np.sum(comp)
            # Filtrar componentes tubulares centrales (30 a 8000 px)
            if 30 <= comp_size <= 8000:
                cy, cx = np.mean(np.where(comp), axis=1)
                # Ubicación central (mediastino / tráquea / carina)
                if (alto * 0.20 < cy < alto * 0.80) and (ancho * 0.30 < cx < ancho * 0.70):
                    airway_mask[z][comp] = 255
                    
    return lung_mask, airway_mask, arr_ct


def guardar_mascara_dicom(ruta_ct_origen, mascara_2d, ruta_salida_dcm, descripcion_serie):
    """
    Clona la cabecera del DICOM original y guarda la máscara como un archivo DICOM 100% válido y compatible.
    """
    try:
        ds = pydicom.dcmread(ruta_ct_origen)
        ds_mask = ds.copy()
        
        # Generar UIDs propios para la nueva serie de máscaras
        ds_mask.SeriesInstanceUID = generate_uid()
        ds_mask.SOPInstanceUID = generate_uid()
        ds_mask.SeriesDescription = descripcion_serie
        ds_mask.Modality = "OT"
        
        # Asignar datos de píxeles en formato uint16
        mask_data = mascara_2d.astype(np.uint16)
        ds_mask.PixelData = mask_data.tobytes()
        
        ds_mask.save_as(ruta_salida_dcm)
    except Exception:
        # Fallback a SimpleITK si pydicom falla
        sitk_img = sitk.GetImageFromArray(mascara_2d.astype(np.uint8))
        sitk.WriteImage(sitk_img, ruta_salida_dcm)


def guardar_resultados_completos(img_sitk, lung_mask, airway_mask, arr_ct, archivos_dcm, dir_salida):
    """
    Guarda los cortes DICOM procesados, las máscaras generadas en DICOM y PNG, y los volúmenes NIfTI 3D.
    """
    dir_ct_out = os.path.join(dir_salida, "dicom_procesado")
    dir_lung_out = os.path.join(dir_salida, "lung_mask")
    dir_airway_out = os.path.join(dir_salida, "airway_mask")
    dir_png_preview = os.path.join(dir_salida, "png_previews")
    
    os.makedirs(dir_ct_out, exist_ok=True)
    os.makedirs(dir_lung_out, exist_ok=True)
    os.makedirs(dir_airway_out, exist_ok=True)
    os.makedirs(dir_png_preview, exist_ok=True)
    
    num_cortes = arr_ct.shape[0]
    print(f"\nGuardando archivos en: {dir_salida}...", flush=True)
    
    # 1. Guardar volúmenes NIfTI 3D
    sitk_lung = sitk.GetImageFromArray(lung_mask)
    sitk_lung.CopyInformation(img_sitk)
    sitk.WriteImage(sitk_lung, os.path.join(dir_salida, "lung_mask_3d.nii.gz"))
    
    sitk_airway = sitk.GetImageFromArray(airway_mask)
    sitk_airway.CopyInformation(img_sitk)
    sitk.WriteImage(sitk_airway, os.path.join(dir_salida, "airway_mask_3d.nii.gz"))
    
    sitk.WriteImage(img_sitk, os.path.join(dir_salida, "tc_volumen_3d.nii.gz"))
    
    # 2. Guardar cortes individuales en dicom_procesado, lung_mask y airway_mask
    for z in range(num_cortes):
        # Nombre de archivo ordenado
        nombre_corte = f"CT_{z+1:04d}.dcm"
        ruta_orig = None
        
        if z < len(archivos_dcm):
            ruta_orig = archivos_dcm[z]
            nombre_corte = os.path.basename(ruta_orig)
            
        nombre_base = os.path.splitext(nombre_corte)[0]
        
        # A) Copiar / Guardar el corte DICOM en dicom_procesado
        ruta_ct_dest = os.path.join(dir_ct_out, nombre_corte)
        if ruta_orig and os.path.exists(ruta_orig):
            shutil.copyfile(ruta_orig, ruta_ct_dest)
        else:
            slice_ct = sitk.GetImageFromArray(arr_ct[z].astype(np.int16))
            sitk.WriteImage(slice_ct, ruta_ct_dest)
            
        # B) Guardar máscara pulmonar en DICOM
        ruta_lung_dcm = os.path.join(dir_lung_out, f"lung_mask_{z+1:04d}.dcm")
        if ruta_orig and os.path.exists(ruta_orig):
            guardar_mascara_dicom(ruta_orig, lung_mask[z], ruta_lung_dcm, "Lung Mask (Deep Learning)")
        else:
            slice_lung = sitk.GetImageFromArray(lung_mask[z])
            sitk.WriteImage(slice_lung, ruta_lung_dcm)
            
        # C) Guardar máscara de vías aéreas en DICOM
        ruta_airway_dcm = os.path.join(dir_airway_out, f"airway_mask_{z+1:04d}.dcm")
        if ruta_orig and os.path.exists(ruta_orig):
            guardar_mascara_dicom(ruta_orig, airway_mask[z], ruta_airway_dcm, "Airway Mask (Trachea/Bronchi)")
        else:
            slice_airway = sitk.GetImageFromArray(airway_mask[z])
            sitk.WriteImage(slice_airway, ruta_airway_dcm)
            
        # D) Guardar preview PNG (para poder ver directamente con visor de fotos de Windows)
        fig_p, ax_p = plt.subplots(1, 3, figsize=(12, 4))
        ax_p[0].imshow(arr_ct[z], cmap='gray', vmin=-1000, vmax=400)
        ax_p[0].set_title(f"TC {nombre_corte}")
        ax_p[0].axis('off')
        
        ax_p[1].imshow(lung_mask[z], cmap='gray')
        ax_p[1].set_title("Mascara Pulmonar")
        ax_p[1].axis('off')
        
        ax_p[2].imshow(airway_mask[z], cmap='gray')
        ax_p[2].set_title("Vias Aereas")
        ax_p[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(dir_png_preview, f"preview_{z+1:04d}.png"), dpi=150)
        plt.close()
        
    print(f"- {num_cortes} cortes guardados en: {dir_ct_out}")
    print(f"- {num_cortes} mascaras de pulmon guardadas en: {dir_lung_out}")
    print(f"- {num_cortes} mascaras de vias aereas guardadas en: {dir_airway_out}")
    print(f"- Previews PNG guardadas en: {dir_png_preview}")
    
    # 3. Generar Mosaico Visual General
    generar_mosaico_preprocesamiento(arr_ct, lung_mask, airway_mask, dir_salida)


def generar_mosaico_preprocesamiento(arr_ct, lung_mask, airway_mask, dir_salida):
    """Crea una imagen resumen mostrando cortes representativos con pulmones y vías aéreas."""
    num_cortes = arr_ct.shape[0]
    paso = max(1, num_cortes // 4)
    indices = list(range(0, num_cortes, paso))[:4]
    
    fig, axes = plt.subplots(len(indices), 3, figsize=(15, 4 * len(indices)))
    if len(indices) == 1:
        axes = np.expand_dims(axes, 0)
        
    for row, z in enumerate(indices):
        # Panel 1: TC Original
        axes[row, 0].imshow(arr_ct[z], cmap='gray', vmin=-1000, vmax=400)
        axes[row, 0].set_title(f"Corte {z+1} - TC Original (Ventana Pulmon)")
        axes[row, 0].axis('off')
        
        # Panel 2: Pulmones
        axes[row, 1].imshow(arr_ct[z], cmap='gray', vmin=-1000, vmax=400)
        axes[row, 1].imshow(np.ma.masked_where(lung_mask[z] == 0, lung_mask[z]), cmap='winter', alpha=0.45)
        axes[row, 1].set_title("Pulmones Segmentados (Deep Learning)")
        axes[row, 1].axis('off')
        
        # Panel 3: Vías Aéreas
        axes[row, 2].imshow(arr_ct[z], cmap='gray', vmin=-1000, vmax=400)
        if np.sum(airway_mask[z]) > 0:
            axes[row, 2].imshow(np.ma.masked_where(airway_mask[z] == 0, airway_mask[z]), cmap='autumn', alpha=0.7)
            axes[row, 2].set_title("Vias Aereas Excluidas (Traquea/Bronquios)")
        else:
            axes[row, 2].set_title("Vias Aereas (Sin componente central)")
        axes[row, 2].axis('off')
        
    plt.suptitle("Preprocesamiento Automatico de TC: Segmentacion Pulmonar y Vias Aereas", fontsize=16)
    plt.tight_layout()
    
    archivo_resumen = os.path.join(dir_salida, "resumen_preprocesamiento.png")
    plt.savefig(archivo_resumen, dpi=200, bbox_inches='tight')
    print(f"- Resumen visual guardado en: {archivo_resumen}", flush=True)
    plt.close()


def preprocesar_tomografia(ruta_entrada, ruta_salida=None):
    if ruta_salida is None:
        nombre_carpeta = os.path.basename(os.path.normpath(ruta_entrada))
        ruta_salida = os.path.join(DATA_DIR, f"preprocesado_{nombre_carpeta}")
        
    os.makedirs(ruta_salida, exist_ok=True)
    
    print("="*60, flush=True)
    print("PREPROCESAMIENTO DE TOMOGRAFIA COMPUTADA (Deep Learning)", flush=True)
    print(f"Entrada: {ruta_entrada}", flush=True)
    print(f"Salida:  {ruta_salida}", flush=True)
    print("="*60, flush=True)
    
    img_sitk, archivos_dcm = leer_serie_dicom(ruta_entrada)
    lung_mask, airway_mask, arr_ct = segmentar_pulmones_y_vias(img_sitk)
    guardar_resultados_completos(img_sitk, lung_mask, airway_mask, arr_ct, archivos_dcm, ruta_salida)
    
    print("\n" + "="*60, flush=True)
    print("PREPROCESAMIENTO FINALIZADO CON EXITO", flush=True)
    print(f"Carpeta de salida: {ruta_salida}")
    print("="*60 + "\n", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Preprocesamiento y Segmentacion Automatica de TC de Pulmon")
    parser.add_argument("--input", "-i", type=str, default=RUTA_DEFAULT_TC, help="Ruta a la carpeta DICOM o archivo tomografico")
    parser.add_argument("--output", "-o", type=str, default=None, help="Ruta de la carpeta de salida (opcional)")
    args = parser.parse_args()
    
    preprocesar_tomografia(args.input, args.output)


if __name__ == "__main__":
    main()
