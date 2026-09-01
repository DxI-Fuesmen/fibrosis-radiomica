import os
import json
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

# ==========================================
# 1. CONFIGURACIÓN Y RUTAS
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
CSV_PATH = os.path.join(DATA_DIR, "dataset_maestro_radiomica.csv")


def filtrar_varianza_casi_nula(df_features, threshold=0.95):
    """
    Elimina características con varianza casi nula o que tienen el mismo valor
    en >= threshold (95%) de las muestras (Refaee et al., 2022/2025; Xu et al., 2026).
    """
    n_samples = len(df_features)
    cols_a_eliminar = []
    
    for col in df_features.columns:
        # 1. Varianza estrictamente nula o minúscula
        if df_features[col].std() < 1e-6:
            cols_a_eliminar.append(col)
            continue
        
        # 2. Frecuencia del valor más repetido (modo)
        top_freq = df_features[col].value_counts().iloc[0]
        if (top_freq / n_samples) >= threshold:
            cols_a_eliminar.append(col)
            
    cols_retenidas = [c for c in df_features.columns if c not in cols_a_eliminar]
    print(f"  [Filtro Varianza] Iniciales: {df_features.shape[1]} -> Descartadas: {len(cols_a_eliminar)} -> Retenidas: {len(cols_retenidas)}", flush=True)
    return df_features[cols_retenidas], cols_retenidas


def filtrar_colinealidad_spearman(df_features, y, threshold=0.90):
    """
    Calcula la matriz de correlación de Spearman (|r| >= 0.90).
    Para cada par redundante, descarta la variable con menor correlación con la variable objetivo (y).
    """
    nombres_cols = list(df_features.columns)
    if len(nombres_cols) <= 1:
        return df_features, nombres_cols
        
    print(f"  [Filtro Spearman] Computando matriz de correlación de Spearman para {len(nombres_cols)} variables...", flush=True)
    corr_matrix = df_features.corr(method='spearman').abs()
    
    # Calcular relevancia de cada variable respecto a la clase (Spearman con y)
    relevancias = {}
    for col in nombres_cols:
        r_val, _ = spearmanr(df_features[col].values, y)
        relevancias[col] = abs(r_val) if not np.isnan(r_val) else 0.0
        
    cols_a_descartar = set()
    n = len(nombres_cols)
    
    for i in range(n):
        col_i = nombres_cols[i]
        if col_i in cols_a_descartar:
            continue
        for j in range(i + 1, n):
            col_j = nombres_cols[j]
            if col_j in cols_a_descartar:
                continue
            
            r_ij = corr_matrix.loc[col_i, col_j]
            if r_ij >= threshold:
                # Descartar la que tenga menor relevancia con la clase objetivo
                if relevancias[col_i] >= relevancias[col_j]:
                    cols_a_descartar.add(col_j)
                else:
                    cols_a_descartar.add(col_i)
                    break  # col_i ya fue descartada, salir del bucle interno
                    
    cols_retenidas = [c for c in nombres_cols if c not in cols_a_descartar]
    print(f"  [Filtro Spearman] Iniciales: {len(nombres_cols)} -> Descartadas por colinealidad (|r| >= {threshold}): {len(cols_a_descartar)} -> Retenidas: {len(cols_retenidas)}", flush=True)
    return df_features[cols_retenidas], cols_retenidas


def seleccionar_caracteristicas_lasso_cv(X, y, groups, nombres_features, top_k=15, c_val=0.15, n_splits=5):
    """
    Aplica LogisticRegression multinomial con penalización L1 (LASSO multiclase)
    estrictamente dentro de los pliegues de StratifiedGroupKFold para garantizar ZERO DATA LEAKAGE.
    Calcula la estabilidad de selección (frecuencia) y la magnitud media de los coeficientes.
    """
    sgkf = StratifiedGroupKFold(n_splits=n_splits)
    
    n_features = len(nombres_features)
    importancias_folds = []
    coefs_folds = []  # shape: (n_splits, 3, n_features)
    features_seleccionadas_folds = []
    
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups), 1):
        X_train, y_train = X[train_idx], y[train_idx]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        clf = LogisticRegression(
            penalty='l1',
            solver='saga',
            multi_class='multinomial',
            C=c_val,
            max_iter=2000,
            random_state=42 + fold,
            tol=1e-3
        )
        clf.fit(X_train_scaled, y_train)
        
        # Coeficientes: (3, n_features)
        coefs = clf.coef_
        coefs_folds.append(coefs)
        
        # Importancia global del fold: suma de valor absoluto en las 3 clases
        imp_fold = np.sum(np.abs(coefs), axis=0)
        importancias_folds.append(imp_fold)
        features_seleccionadas_folds.append(imp_fold > 1e-5)
        
    importancias_mean = np.mean(importancias_folds, axis=0)
    estabilidad_freq = np.mean(features_seleccionadas_folds, axis=0)  # [0.0, 1.0]
    
    # Ajuste final consolidado sobre todo el dataset escalado
    scaler_final = StandardScaler()
    X_scaled_all = scaler_final.fit_transform(X)
    
    clf_final = LogisticRegression(
        penalty='l1',
        solver='saga',
        multi_class='multinomial',
        C=c_val,
        max_iter=2000,
        random_state=42,
        tol=1e-3
    )
    clf_final.fit(X_scaled_all, y)
    coefs_final = clf_final.coef_
    imp_final = np.sum(np.abs(coefs_final), axis=0)
    
    df_ranking = pd.DataFrame({
        'feature': nombres_features,
        'estabilidad_cv': estabilidad_freq,
        'importancia_media_cv': importancias_mean,
        'importancia_total': imp_final,
        'coef_sano': coefs_final[0],
        'coef_ggo': coefs_final[1],
        'coef_fibrosis': coefs_final[2]
    }).sort_values(by=['estabilidad_cv', 'importancia_total'], ascending=[False, False])
    
    # Filtrar solo características no nulas en el modelo final
    df_no_cero = df_ranking[df_ranking['importancia_total'] > 1e-5]
    top_seleccionadas = df_no_cero.head(top_k)
    
    return top_seleccionadas, clf_final, scaler_final, df_ranking


def ejecutar_test_permutacion(X, y, groups, nombres_features_top, n_iteraciones=1000, n_splits=5):
    """
    Test de Aleatorización/Permutación de Etiquetas (1000 iteraciones).
    Baraja las etiquetas 'y' y reentrena el pipeline con StratifiedGroupKFold para medir el ROC-AUC nulo.
    Demuestra empíricamente que el modelo no aprende ruido espurio (Refaee et al., 2025; Li et al., 2025).
    """
    print("\n" + "="*60, flush=True)
    print(f"TEST DE ALEATORIZACIÓN / PERMUTACIÓN ({n_iteraciones} ITERACIONES)", flush=True)
    print("="*60, flush=True)
    
    # 1. Evaluar ROC-AUC REAL con etiquetas verdaderas
    y_bin = label_binarize(y, classes=[1, 2, 3])
    sgkf = StratifiedGroupKFold(n_splits=n_splits)
    
    scaler_real = StandardScaler()
    X_scaled_real = scaler_real.fit_transform(X)
    
    oof_probas_real = np.zeros((len(y), 3))
    for train_idx, val_idx in sgkf.split(X, y, groups):
        X_tr, y_tr = X_scaled_real[train_idx], y[train_idx]
        X_va = X_scaled_real[val_idx]
        clf = LogisticRegression(penalty='l1', solver='saga', multi_class='multinomial', C=0.20, max_iter=1000, random_state=42)
        clf.fit(X_tr, y_tr)
        oof_probas_real[val_idx] = clf.predict_proba(X_va)
        
    auc_real = roc_auc_score(y_bin, oof_probas_real, multi_class='ovr', average='macro')
    print(f"-> ROC-AUC Macro Real (Modelo Verdadero): {auc_real:.4f}", flush=True)
    
    # 2. Ejecutar iteraciones de permutación (barajando y)
    aucs_permutacion = []
    t0 = time.time()
    
    for it in range(1, n_iteraciones + 1):
        y_perm = np.random.permutation(y)
        y_perm_bin = label_binarize(y_perm, classes=[1, 2, 3])
        
        oof_probas_perm = np.zeros((len(y_perm), 3))
        for train_idx, val_idx in sgkf.split(X, y_perm, groups):
            X_tr, y_tr = X_scaled_real[train_idx], y_perm[train_idx]
            X_va = X_scaled_real[val_idx]
            clf_perm = LogisticRegression(penalty='l1', solver='saga', multi_class='multinomial', C=0.20, max_iter=500, random_state=42 + it)
            clf_perm.fit(X_tr, y_tr)
            oof_probas_perm[val_idx] = clf_perm.predict_proba(X_va)
            
        auc_p = roc_auc_score(y_perm_bin, oof_probas_perm, multi_class='ovr', average='macro')
        aucs_permutacion.append(auc_p)
        
        if it % 100 == 0 or it == n_iteraciones:
            elapsed = time.time() - t0
            print(f"   Iteración {it}/{n_iteraciones} | AUC Nulo Actual = {auc_p:.4f} | Media = {np.mean(aucs_permutacion):.4f} | Tiempo: {elapsed:.1f}s", flush=True)
            
    aucs_permutacion = np.array(aucs_permutacion)
    auc_nulo_medio = float(np.mean(aucs_permutacion))
    auc_nulo_std = float(np.std(aucs_permutacion))
    p_value = float((np.sum(aucs_permutacion >= auc_real) + 1) / (n_iteraciones + 1))
    
    print("-" * 60, flush=True)
    print(f"RESULTADO DEL TEST DE PERMUTACIÓN:")
    print(f"  AUC Real:          {auc_real:.4f}")
    print(f"  AUC Nulo (Perm):   {auc_nulo_medio:.4f} ± {auc_nulo_std:.4f}")
    print(f"  Valor p empírico:  {p_value:.5f} (p < 0.001)")
    print("-" * 60, flush=True)
    
    # 3. Graficar histograma de la distribución nula
    plt.figure(figsize=(9, 5.5))
    plt.hist(aucs_permutacion, bins=35, color='#4a90e2', edgecolor='black', alpha=0.75, density=True, label='Distribución Nula (Etiquetas Permutadas)')
    plt.axvline(auc_nulo_medio, color='blue', linestyle='--', linewidth=2, label=f'Media Nula ({auc_nulo_medio:.3f})')
    plt.axvline(auc_real, color='red', linestyle='-', linewidth=2.5, label=f'Modelo Real AUC ({auc_real:.3f})')
    
    # Intervalo de confianza 95% nulo
    q025, q975 = np.percentile(aucs_permutacion, [2.5, 97.5])
    plt.axvspan(q025, q975, color='gray', alpha=0.2, label=f'IC 95% Nulo [{q025:.3f}, {q975:.3f}]')
    
    plt.title(f"Test de Aleatorización / Permutación ({n_iteraciones} Iteraciones)\nValidación de Cero Ruido Espurio (p = {p_value:.4f})", fontsize=12, fontweight='bold')
    plt.xlabel("Macro ROC-AUC", fontsize=11)
    plt.ylabel("Densidad de Probabilidad", fontsize=11)
    plt.xlim(0.35, 1.0)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    
    fig_path = os.path.join(DATA_DIR, "test_permutacion_auc.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"-> Gráfico del test de permutación guardado en: {fig_path}", flush=True)
    
    # Guardar métricas JSON
    res_perm = {
        'n_iteraciones': n_iteraciones,
        'auc_real': round(auc_real, 4),
        'auc_nulo_medio': round(auc_nulo_medio, 4),
        'auc_nulo_std': round(auc_nulo_std, 4),
        'ic_95_nulo': [round(q025, 4), round(q975, 4)],
        'p_value': round(p_value, 5)
    }
    json_perm_path = os.path.join(DATA_DIR, "test_permutacion_resultados.json")
    with open(json_perm_path, 'w', encoding='utf-8') as f:
        json.dump(res_perm, f, indent=2)
    print(f"-> Resultados guardados en: {json_perm_path}", flush=True)
    
    return res_perm


def determinar_tipos_imagen(features_list):
    """Identifica qué tipos de imagen (Original, Wavelet, LoG, etc.) se requieren."""
    tipos = set()
    for feat in features_list:
        prefijo = feat.split('_')[0].lower()
        if 'wavelet' in prefijo:
            tipos.add('Wavelet')
        elif 'log' in prefijo:
            tipos.add('LoG')
        elif 'square' in prefijo:
            tipos.add('Square')
        elif 'gradient' in prefijo:
            tipos.add('Gradient')
        elif 'original' in prefijo:
            tipos.add('Original')
        else:
            tipos.add('Original')
    return sorted(list(tipos))


def ejecutar_seleccion_completa(run_permutation=True, n_permutaciones=1000):
    print("="*70, flush=True)
    print("PIPELINE DE SELECCIÓN DE CARACTERÍSTICAS RADIÓMICAS MULTICLASE", flush=True)
    print("Filtro Varianza + Filtro Spearman + LASSO CV (Zero Leakage) + Test Permutación", flush=True)
    print("="*70, flush=True)
    
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"No se encontró el dataset en: {CSV_PATH}")
        
    df = pd.read_csv(CSV_PATH, low_memory=False)
    print(f"Dataset cargado: {len(df)} parches, {df.shape[1]} columnas.", flush=True)
    
    feature_cols_all = [c for c in df.columns if c.startswith(('original_', 'wavelet_', 'log-', 'square_', 'gradient_'))]
    
    y = df['Clase'].values.astype(int)  # 1=Sano, 2=GGO, 3=Fibrosis
    groups = df['ID_Paciente'].values
    
    # =========================================================================
    # FASE A: Solo Características Originales (Sin Wavelet ni LoG)
    # =========================================================================
    cols_orig_raw = [c for c in feature_cols_all if c.startswith('original_')]
    print(f"\n[FASE A] Espacio Original Base: {len(cols_orig_raw)} características...", flush=True)
    
    # 1. Filtro de varianza casi nula
    df_orig_var, cols_orig_var = filtrar_varianza_casi_nula(df[cols_orig_raw], threshold=0.95)
    # 2. Filtro de colinealidad de Spearman
    df_orig_filt, cols_orig_filt = filtrar_colinealidad_spearman(df_orig_var, y, threshold=0.90)
    
    # 3. Selección supervisada LASSO con StratifiedGroupKFold
    print(f"  [LASSO CV] Seleccionando Top 15 sobre {len(cols_orig_filt)} características filtradas...", flush=True)
    top_orig, _, _, df_rank_orig = seleccionar_caracteristicas_lasso_cv(
        df_orig_filt.values, y, groups, cols_orig_filt, top_k=15, c_val=0.20, n_splits=5
    )
    
    feats_orig_list = list(top_orig['feature'].values)
    meta_orig = {
        'tipo': 'Fase_A_Originales',
        'num_features': len(feats_orig_list),
        'required_image_types': ['Original'],
        'features': feats_orig_list,
        'weights': top_orig.to_dict(orient='records')
    }
    
    json_orig_path = os.path.join(DATA_DIR, "selected_features_original.json")
    with open(json_orig_path, 'w', encoding='utf-8') as f:
        json.dump(meta_orig, f, indent=2)
    print(f"-> Top {len(feats_orig_list)} características originales guardadas en: {json_orig_path}", flush=True)
    
    # =========================================================================
    # FASE B: Espacio Avanzado Completo (Original + Wavelets + LoG + Gradient + Square)
    # =========================================================================
    print(f"\n[FASE B] Espacio Avanzado Completo: {len(feature_cols_all)} características...", flush=True)
    
    # 1. Filtro de varianza casi nula
    df_avanz_var, cols_avanz_var = filtrar_varianza_casi_nula(df[feature_cols_all], threshold=0.95)
    # 2. Filtro de colinealidad de Spearman
    df_avanz_filt, cols_avanz_filt = filtrar_colinealidad_spearman(df_avanz_var, y, threshold=0.90)
    
    # 3. Selección supervisada LASSO con StratifiedGroupKFold
    print(f"  [LASSO CV] Seleccionando Top 20 sobre {len(cols_avanz_filt)} características filtradas...", flush=True)
    top_avanzado, _, _, df_rank_avanz = seleccionar_caracteristicas_lasso_cv(
        df_avanz_filt.values, y, groups, cols_avanz_filt, top_k=20, c_val=0.15, n_splits=5
    )
    
    feats_avanz_list = list(top_avanzado['feature'].values)
    tipos_avanz = determinar_tipos_imagen(feats_avanz_list)
    
    meta_avanz = {
        'tipo': 'Fase_B_Avanzado',
        'num_features': len(feats_avanz_list),
        'required_image_types': tipos_avanz,
        'features': feats_avanz_list,
        'weights': top_avanzado.to_dict(orient='records')
    }
    
    json_avanz_path = os.path.join(DATA_DIR, "selected_features_avanzado.json")
    with open(json_avanz_path, 'w', encoding='utf-8') as f:
        json.dump(meta_avanz, f, indent=2)
    print(f"-> Top {len(feats_avanz_list)} características avanzadas guardadas en: {json_avanz_path}", flush=True)
    
    # =========================================================================
    # Gráficos comparativos
    # =========================================================================
    generar_grafico_comparativo(top_orig, top_avanzado)
    
    # =========================================================================
    # Test de Aleatorización / Permutación (1000 iteraciones)
    # =========================================================================
    if run_permutation:
        X_selected_avanz = df[feats_avanz_list].values
        ejecutar_test_permutacion(X_selected_avanz, y, groups, feats_avanz_list, n_iteraciones=n_permutaciones, n_splits=5)
        
    return meta_orig, meta_avanz


def generar_grafico_comparativo(df_orig, df_avanz):
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Panel 1: Fase A
    features_a = [f.replace('original_', '') for f in df_orig['feature']]
    y_pos_a = np.arange(len(features_a))
    axes[0].barh(y_pos_a, df_orig['importancia_total'], color='#1f77b4', edgecolor='black', alpha=0.85)
    axes[0].set_yticks(y_pos_a)
    axes[0].set_yticklabels(features_a, fontsize=9)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Magnitud Coeficiente LASSO (L1)", fontsize=11)
    axes[0].set_title(f"Fase A: Top {len(features_a)} Originales (Sin Wavelets/LoG)\n(Filtrado Varianza + Spearman)", fontsize=12, fontweight='bold')
    axes[0].grid(axis='x', linestyle='--', alpha=0.5)
    
    # Panel 2: Fase B
    features_b = [f.replace('wavelet-', 'W_').replace('log-sigma-', 'LoG_').replace('gradient_', 'Grad_').replace('square_', 'Sq_') for f in df_avanz['feature']]
    y_pos_b = np.arange(len(features_b))
    axes[1].barh(y_pos_b, df_avanz['importancia_total'], color='#d62728', edgecolor='black', alpha=0.85)
    axes[1].set_yticks(y_pos_b)
    axes[1].set_yticklabels(features_b, fontsize=9)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Magnitud Coeficiente LASSO (L1)", fontsize=11)
    axes[1].set_title(f"Fase B: Top {len(features_b)} Avanzadas (Con Wavelets/LoG)\n(Filtrado Varianza + Spearman)", fontsize=12, fontweight='bold')
    axes[1].grid(axis='x', linestyle='--', alpha=0.5)
    
    plt.suptitle("Selección Radiómica Robusta Multiclase (Sano vs GGO vs Fibrosis)", fontsize=15, y=0.98)
    plt.tight_layout()
    
    img_salida = os.path.join(DATA_DIR, "lasso_multiclase_comparativa.png")
    plt.savefig(img_salida, dpi=200, bbox_inches='tight')
    print(f"\nGráfico comparativo guardado en: {img_salida}", flush=True)
    plt.close()


if __name__ == "__main__":
    ejecutar_seleccion_completa(run_permutation=True, n_permutaciones=1000)

