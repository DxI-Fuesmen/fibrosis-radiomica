import os
import json
import time
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, balanced_accuracy_score, confusion_matrix, roc_curve

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")

# ==========================================
# 1. CONFIGURACIÓN Y RUTAS
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

CSV_PATH = os.path.join(DATA_DIR, "dataset_maestro_radiomica.csv")
JSON_ORIG = os.path.join(DATA_DIR, "selected_features_original.json")
JSON_AVANZ = os.path.join(DATA_DIR, "selected_features_avanzado.json")

CLASES = [1, 2, 3]
NOMBRES_CLASES = ['Sano', 'GGO', 'Fibrosis']


def inicializar_modelos():
    """Define los modelos a comparar en el benchmark."""
    modelos = {
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=2, random_state=42, n_jobs=-1
        ),
        'Extra Trees': ExtraTreesClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=2, random_state=42, n_jobs=-1
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=150, max_depth=6, learning_rate=0.08, eval_metric='mlogloss',
            random_state=42, n_jobs=-1
        ),
        'LightGBM': lgb.LGBMClassifier(
            n_estimators=150, max_depth=6, learning_rate=0.08, random_state=42,
            n_jobs=-1, verbose=-1
        ),
        'SVM (RBF)': SVC(
            C=3.0, kernel='rbf', gamma='scale', probability=True, random_state=42
        ),
        'Logistic Regression (L1)': LogisticRegression(
            penalty='l1', solver='saga', multi_class='multinomial', C=0.2,
            max_iter=1000, random_state=42
        )
    }
    return modelos


def evaluar_modelo_cv(nombre_modelo, modelo, X, y, groups, n_splits=5):
    """
    Ejecuta StratifiedGroupKFold para un modelo y calcula métricas completas, tiempos y predicciones OOF.
    """
    sgkf = StratifiedGroupKFold(n_splits=n_splits)
    
    y_true_all = np.zeros(len(y), dtype=int)
    y_pred_all = np.zeros(len(y), dtype=int)
    y_proba_all = np.zeros((len(y), 3), dtype=float)
    
    tiempos_inferencia_ms = []
    tiempos_entrenamiento_s = []
    
    # Mapear clases a 0, 1, 2 para XGBoost/LightGBM
    y_zero_based = y - 1
    
    for train_idx, val_idx in sgkf.split(X, y_zero_based, groups):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y_zero_based[train_idx], y_zero_based[val_idx]
        
        # Escalamiento de variables
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # 1. Medir tiempo de entrenamiento
        t0 = time.time()
        modelo.fit(X_train_scaled, y_train)
        tiempos_entrenamiento_s.append(time.time() - t0)
        
        # 2. Medir tiempo de inferencia
        t_inf0 = time.time()
        probas = modelo.predict_proba(X_val_scaled)
        preds = np.argmax(probas, axis=1)
        dt_inf = (time.time() - t_inf0) / len(X_val) * 1000.0  # ms por parche
        tiempos_inferencia_ms.append(dt_inf)
        
        y_true_all[val_idx] = y_val
        y_pred_all[val_idx] = preds
        y_proba_all[val_idx] = probas
        
    # Binarizar para ROC-AUC One-vs-Rest
    y_true_bin = label_binarize(y_true_all, classes=[0, 1, 2])
    auc_macro = roc_auc_score(y_true_bin, y_proba_all, average='macro', multi_class='ovr')
    auc_sano = roc_auc_score(y_true_bin[:, 0], y_proba_all[:, 0])
    auc_ggo = roc_auc_score(y_true_bin[:, 1], y_proba_all[:, 1])
    auc_fib = roc_auc_score(y_true_bin[:, 2], y_proba_all[:, 2])
    
    acc = accuracy_score(y_true_all, y_pred_all)
    bal_acc = balanced_accuracy_score(y_true_all, y_pred_all)
    f1_macro = f1_score(y_true_all, y_pred_all, average='macro')
    
    cm = confusion_matrix(y_true_all, y_pred_all)
    
    tiempo_inf_medio = np.mean(tiempos_inferencia_ms)
    tiempo_ent_medio = np.mean(tiempos_entrenamiento_s)
    tiempo_corte_ms = tiempo_inf_medio * 500
    
    resultados = {
        'Modelo': nombre_modelo,
        'ROC_AUC_Macro': auc_macro,
        'AUC_Sano': auc_sano,
        'AUC_GGO': auc_ggo,
        'AUC_Fibrosis': auc_fib,
        'Accuracy': acc,
        'Balanced_Acc': bal_acc,
        'F1_Macro': f1_macro,
        'Tiempo_Inf_ms_parche': tiempo_inf_medio,
        'Tiempo_Corte_Estimado_ms': tiempo_corte_ms,
        'Tiempo_Entrenamiento_s': tiempo_ent_medio,
        'Confusion_Matrix': cm,
        'y_true': y_true_all,
        'y_proba': y_proba_all
    }
    return resultados


def calcular_umbrales_optimos_youden(y_true, y_proba):
    """
    Calcula el umbral óptimo de decisión para cada clase mediante el Índice de Youden (J = Sens + Espec - 1)
    sobre las predicciones Out-of-Fold del conjunto de entrenamiento.
    """
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
    umbrales_optimos = {}
    puntos_youden = {}
    
    for c_idx, nombre in enumerate(NOMBRES_CLASES):
        fpr, tpr, thresholds = roc_curve(y_true_bin[:, c_idx], y_proba[:, c_idx])
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        
        best_thresh = float(thresholds[best_idx])
        # Asegurar un rango razonable [0.05, 0.95]
        best_thresh = max(0.05, min(0.95, best_thresh))
        
        umbrales_optimos[nombre.lower()] = round(best_thresh, 4)
        puntos_youden[nombre] = {
            'fpr': float(fpr[best_idx]),
            'tpr': float(tpr[best_idx]),
            'threshold': best_thresh,
            'j_score': float(j_scores[best_idx])
        }
        
    print("\n" + "="*60, flush=True)
    print("UMBRALES ÓPTIMOS DE DECISIÓN (ÍNDICE DE YOUDEN EN OOF):", flush=True)
    for nombre in NOMBRES_CLASES:
        p = puntos_youden[nombre]
        print(f"  • {nombre:<10}: Umbral = {p['threshold']:.4f} | Sens = {p['tpr']*100:.1f}% | Espec = {(1-p['fpr'])*100:.1f}% | J = {p['j_score']:.4f}", flush=True)
    print("="*60, flush=True)
    
    # Graficar curvas ROC con puntos de Youden
    plt.figure(figsize=(8, 6))
    colores_roc = {'Sano': '#2ca02c', 'GGO': '#ff7f0e', 'Fibrosis': '#d62728'}
    for c_idx, nombre in enumerate(NOMBRES_CLASES):
        fpr, tpr, _ = roc_curve(y_true_bin[:, c_idx], y_proba[:, c_idx])
        auc_val = roc_auc_score(y_true_bin[:, c_idx], y_proba[:, c_idx])
        plt.plot(fpr, tpr, color=colores_roc[nombre], lw=2.5, label=f"{nombre} (AUC = {auc_val:.4f})")
        # Marcar punto de Youden
        py = puntos_youden[nombre]
        plt.plot(py['fpr'], py['tpr'], marker='o', markersize=9, color=colores_roc[nombre], markeredgecolor='black',
                 label=f"Youden {nombre}: θ={py['threshold']:.2f}")
        
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.6)
    plt.xlabel("1 - Especificidad (FPR)", fontsize=11)
    plt.ylabel("Sensibilidad (TPR)", fontsize=11)
    plt.title("Curvas ROC y Puntos de Decisión Óptimos (Índice de Youden)", fontsize=13, fontweight='bold')
    plt.legend(loc='lower right', fontsize=9.5)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    fig_youden_path = os.path.join(DATA_DIR, "umbrales_optimos_youden.png")
    plt.savefig(fig_youden_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"-> Gráfico de curvas ROC con umbrales de Youden guardado en: {fig_youden_path}", flush=True)
    
    return umbrales_optimos, puntos_youden


def optimizar_pesos_decision_grid(y_true, y_proba):
    """
    Realiza una búsqueda en cuadrícula fina sobre los pesos de decisión para maximizar el F1-score macro multiclase.
    Predicción = argmax( y_proba / umbrales )
    """
    best_f1 = -1.0
    best_weights = [1.0, 1.0, 1.0]
    
    w_sano_vals = np.linspace(0.6, 1.4, 9)
    w_ggo_vals = np.linspace(0.6, 1.4, 9)
    w_fib_vals = np.linspace(0.6, 1.4, 9)
    
    for ws in w_sano_vals:
        for wg in w_ggo_vals:
            for wf in w_fib_vals:
                w_vec = np.array([ws, wg, wf])
                probas_weighted = y_proba * w_vec
                preds = np.argmax(probas_weighted, axis=1)
                score = f1_score(y_true, preds, average='macro')
                if score > best_f1:
                    best_f1 = score
                    best_weights = [round(float(ws), 2), round(float(wg), 2), round(float(wf), 2)]
                    
    print(f"-> Pesos Óptimos de Ponderación Multiclase: Sano={best_weights[0]}, GGO={best_weights[1]}, Fibrosis={best_weights[2]} (Macro F1 = {best_f1*100:.2f}%)", flush=True)
    return best_weights


def ejecutar_benchmark():
    print("="*60, flush=True)
    print("BENCHMARK MULTIMODELO RADIÓMICO (Group 5-Fold Cross Validation)", flush=True)
    print("="*60, flush=True)
    
    df = pd.read_csv(CSV_PATH, low_memory=False)
    y = df['Clase'].values.astype(int)
    groups = df['ID_Paciente'].values
    
    with open(JSON_ORIG, 'r', encoding='utf-8') as f:
        meta_orig = json.load(f)
    with open(JSON_AVANZ, 'r', encoding='utf-8') as f:
        meta_avanz = json.load(f)
        
    cols_orig = meta_orig['features']
    cols_avanz = meta_avanz['features']
    
    X_orig = df[cols_orig].values
    X_avanz = df[cols_avanz].values
    
    res_fase_a = []
    res_fase_b = []
    
    # ----------------------------------------------------
    # FASE A: Características Originales (Sin Wavelets ni LoG)
    # ----------------------------------------------------
    print(f"\n---> Evaluando Modelos en FASE A ({len(cols_orig)} características originales)...", flush=True)
    for nombre, clf in inicializar_modelos().items():
        print(f"     Entrenando {nombre}...", flush=True)
        r = evaluar_modelo_cv(f"{nombre} (Original)", clf, X_orig, y, groups)
        res_fase_a.append(r)
        print(f"       -> ROC-AUC: {r['ROC_AUC_Macro']:.4f} | F1: {r['F1_Macro']:.4f} | Inf: {r['Tiempo_Inf_ms_parche']:.3f} ms/p", flush=True)
        
    # ----------------------------------------------------
    # FASE B: Características Avanzadas (Con Wavelets y LoG)
    # ----------------------------------------------------
    print(f"\n---> Evaluando Modelos en FASE B ({len(cols_avanz)} características avanzadas)...", flush=True)
    for nombre, clf in inicializar_modelos().items():
        print(f"     Entrenando {nombre}...", flush=True)
        r = evaluar_modelo_cv(f"{nombre} (Avanzado)", clf, X_avanz, y, groups)
        res_fase_b.append(r)
        print(f"       -> ROC-AUC: {r['ROC_AUC_Macro']:.4f} | F1: {r['F1_Macro']:.4f} | Inf: {r['Tiempo_Inf_ms_parche']:.3f} ms/p", flush=True)
        
    todos_resultados = res_fase_a + res_fase_b
    
    # Crear DataFrame resumen
    filas_tabla = []
    for r in todos_resultados:
        filas_tabla.append({
            'Modelo': r['Modelo'],
            'ROC_AUC_Macro': round(r['ROC_AUC_Macro'], 4),
            'AUC_Sano': round(r['AUC_Sano'], 4),
            'AUC_GGO': round(r['AUC_GGO'], 4),
            'AUC_Fibrosis': round(r['AUC_Fibrosis'], 4),
            'Accuracy': round(r['Accuracy'] * 100, 2),
            'Balanced_Acc': round(r['Balanced_Acc'] * 100, 2),
            'F1_Macro': round(r['F1_Macro'] * 100, 2),
            'Tiempo_Inf_ms': round(r['Tiempo_Inf_ms_parche'], 4),
            'Tiempo_Corte_ms': round(r['Tiempo_Corte_Estimado_ms'], 1)
        })
    df_tabla = pd.DataFrame(filas_tabla).sort_values(by='ROC_AUC_Macro', ascending=False)
    
    tabla_csv_path = os.path.join(DATA_DIR, "benchmark_resultados.csv")
    df_tabla.to_csv(tabla_csv_path, index=False)
    
    print("\n" + "="*80, flush=True)
    print("RESUMEN DEL BENCHMARK MULTIMODELO (Ordenado por ROC-AUC Macro)", flush=True)
    print("="*80, flush=True)
    print(df_tabla.to_string(index=False), flush=True)
    print("="*80 + "\n", flush=True)
    
    # ----------------------------------------------------
    # Selección y Optimización de Umbrales del Mejor Modelo
    # ----------------------------------------------------
    mejor_fila = df_tabla.iloc[0]
    mejor_nombre = mejor_fila['Modelo']
    es_avanzado = 'Avanzado' in mejor_nombre
    
    # Buscar el resultado OOF correspondiente al mejor modelo
    mejor_res = next(r for r in todos_resultados if r['Modelo'] == mejor_nombre)
    umbrales_youden, puntos_youden = calcular_umbrales_optimos_youden(mejor_res['y_true'], mejor_res['y_proba'])
    pesos_optimos = optimizar_pesos_decision_grid(mejor_res['y_true'], mejor_res['y_proba'])
    
    # Guardar archivo de umbrales óptimos
    config_umbrales = {
        'modelo': mejor_nombre,
        'umbrales_youden': umbrales_youden,
        'puntos_detalle': puntos_youden,
        'pesos_decision_optimos': {
            'sano': pesos_optimos[0],
            'ggo': pesos_optimos[1],
            'fibrosis': pesos_optimos[2]
        }
    }
    json_umbrales_path = os.path.join(DATA_DIR, "optimal_thresholds.json")
    with open(json_umbrales_path, 'w', encoding='utf-8') as f:
        json.dump(config_umbrales, f, indent=2)
    print(f"-> Configuración de umbrales óptimos guardada en: {json_umbrales_path}", flush=True)
    
    # ----------------------------------------------------
    # Entrenar y Guardar el Mejor Modelo Final sobre Todo el Dataset
    # ----------------------------------------------------
    print(f"--> MEJOR MODELO: {mejor_nombre} (ROC-AUC: {mejor_fila['ROC_AUC_Macro']:.4f})", flush=True)
    
    cols_ganadoras = cols_avanz if es_avanzado else cols_orig
    meta_ganador = meta_avanz if es_avanzado else meta_orig
    X_ganador = df[cols_ganadoras].values
    
    scaler_final = StandardScaler()
    X_scaled = scaler_final.fit_transform(X_ganador)
    
    # Instanciar el mejor modelo
    if 'Random Forest' in mejor_nombre:
        mejor_clf = RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=2, random_state=42, n_jobs=-1)
    elif 'XGBoost' in mejor_nombre:
        mejor_clf = xgb.XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.08, eval_metric='mlogloss', random_state=42, n_jobs=-1)
    elif 'LightGBM' in mejor_nombre:
        mejor_clf = lgb.LGBMClassifier(n_estimators=150, max_depth=6, learning_rate=0.08, random_state=42, n_jobs=-1, verbose=-1)
    elif 'SVM' in mejor_nombre:
        mejor_clf = SVC(C=3.0, kernel='rbf', gamma='scale', probability=True, random_state=42)
    elif 'Logistic Regression' in mejor_nombre:
        mejor_clf = LogisticRegression(penalty='l1', solver='saga', multi_class='multinomial', C=0.2, max_iter=1000, random_state=42)
    else:
        mejor_clf = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
        
    mejor_clf.fit(X_scaled, y - 1)  # 0=Sano, 1=GGO, 2=Fibrosis
    
    # Guardar modelo entrenado con su escalador, metadatos y UMBRALES ÓPTIMOS
    artefacto_guardado = {
        'model': mejor_clf,
        'scaler': scaler_final,
        'model_name': mejor_nombre,
        'features': cols_ganadoras,
        'required_image_types': meta_ganador['required_image_types'],
        'classes': [0, 1, 2],
        'class_names': NOMBRES_CLASES,
        'optimal_thresholds': umbrales_youden,
        'optimal_weights': {
            'sano': pesos_optimos[0],
            'ggo': pesos_optimos[1],
            'fibrosis': pesos_optimos[2]
        }
    }
    
    modelo_path = os.path.join(MODELS_DIR, "mejor_modelo_multiclase.pkl")
    joblib.dump(artefacto_guardado, modelo_path)
    print(f"Mejor modelo guardado exitosamente en: {modelo_path}", flush=True)
    
    # Generar visualizaciones completas del benchmark
    generar_visualizaciones_benchmark(todos_resultados, df_tabla)
    
    return df_tabla, artefacto_guardado


def generar_visualizaciones_benchmark(resultados, df_tabla):
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.25)
    
    # 1. Gráfico de Barras: ROC-AUC Macro
    ax1 = fig.add_subplot(gs[0, 0])
    df_sort_auc = df_tabla.sort_values(by='ROC_AUC_Macro', ascending=True)
    colores = ['#1f77b4' if 'Original' in m else '#d62728' for m in df_sort_auc['Modelo']]
    bars = ax1.barh(df_sort_auc['Modelo'], df_sort_auc['ROC_AUC_Macro'], color=colores, alpha=0.85, edgecolor='black')
    ax1.set_xlim(0.80, 1.0)
    ax1.set_xlabel("ROC-AUC Macro (Group 5-Fold CV)", fontsize=11)
    ax1.set_title("Comparación de Rendimiento Diagnóstico (ROC-AUC)", fontsize=12, fontweight='bold')
    ax1.grid(axis='x', linestyle='--', alpha=0.5)
    for bar in bars:
        w = bar.get_width()
        ax1.text(w - 0.03, bar.get_y() + bar.get_height()/2, f"{w:.4f}", ha='left', va='center', color='white', fontweight='bold', fontsize=9)
        
    # 2. Gráfico de Barras: Tiempo de Inferencia por Corte (ms)
    ax2 = fig.add_subplot(gs[0, 1])
    df_sort_time = df_tabla.sort_values(by='Tiempo_Corte_ms', ascending=False)
    colores_t = ['#1f77b4' if 'Original' in m else '#d62728' for m in df_sort_time['Modelo']]
    bars2 = ax2.barh(df_sort_time['Modelo'], df_sort_time['Tiempo_Corte_ms'], color=colores_t, alpha=0.85, edgecolor='black')
    ax2.set_xlabel("Tiempo Estimado por Corte (ms)", fontsize=11)
    ax2.set_title("Eficiencia Computacional (Tiempo por Corte)", fontsize=12, fontweight='bold')
    ax2.grid(axis='x', linestyle='--', alpha=0.5)
    
    # 3. Trade-off ROC-AUC vs Tiempo de Inferencia
    ax3 = fig.add_subplot(gs[0, 2])
    for r in resultados:
        color = '#1f77b4' if 'Original' in r['Modelo'] else '#d62728'
        marker = 'o' if 'Random Forest' in r['Modelo'] else ('s' if 'XGBoost' in r['Modelo'] else ('^' if 'SVM' in r['Modelo'] else 'D'))
        ax3.scatter(r['Tiempo_Inf_ms_parche'], r['ROC_AUC_Macro'], color=color, s=140, marker=marker, edgecolors='black', label=r['Modelo'])
        ax3.text(r['Tiempo_Inf_ms_parche'] * 1.05, r['ROC_AUC_Macro'] - 0.002, r['Modelo'].split(' ')[0], fontsize=8)
    ax3.set_xlabel("Tiempo Inferencia (ms por parche)", fontsize=11)
    ax3.set_ylabel("ROC-AUC Macro", fontsize=11)
    ax3.set_title("Trade-off: ROC-AUC vs Latencia", fontsize=12, fontweight='bold')
    ax3.grid(True, linestyle='--', alpha=0.5)
    
    # 4. Matriz de Confusión del Mejor Modelo
    mejor_res = max(resultados, key=lambda x: x['ROC_AUC_Macro'])
    ax4 = fig.add_subplot(gs[1, 0])
    sns.heatmap(mejor_res['Confusion_Matrix'], annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=NOMBRES_CLASES, yticklabels=NOMBRES_CLASES, ax=ax4)
    ax4.set_xlabel("Predicción del Modelo", fontsize=11)
    ax4.set_ylabel("Ground Truth (Anotación Real)", fontsize=11)
    ax4.set_title(f"Matriz de Confusión: {mejor_res['Modelo']}", fontsize=12, fontweight='bold')
    
    # 5. Curvas ROC Multiclase (One-vs-Rest) del Mejor Modelo
    ax5 = fig.add_subplot(gs[1, 1:])
    y_true_bin = label_binarize(mejor_res['y_true'], classes=[0, 1, 2])
    colores_roc = ['#2ca02c', '#ff7f0e', '#d62728']
    for c_idx in range(3):
        fpr, tpr, _ = roc_curve(y_true_bin[:, c_idx], mejor_res['y_proba'][:, c_idx])
        auc_val = roc_auc_score(y_true_bin[:, c_idx], mejor_res['y_proba'][:, c_idx])
        ax5.plot(fpr, tpr, color=colores_roc[c_idx], lw=2.5, label=f"{NOMBRES_CLASES[c_idx]} (AUC = {auc_val:.4f})")
    ax5.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.6)
    ax5.set_xlabel("Tasa de Falsos Positivos (1 - Especificidad)", fontsize=11)
    ax5.set_ylabel("Tasa de Verdaderos Positivos (Sensibilidad)", fontsize=11)
    ax5.set_title(f"Curvas ROC One-vs-Rest (Macro AUC: {mejor_res['ROC_AUC_Macro']:.4f})", fontsize=12, fontweight='bold')
    ax5.legend(loc='lower right', fontsize=11)
    ax5.grid(True, linestyle='--', alpha=0.5)
    
    plt.suptitle("BENCHMARK MULTIMODELO RADIÓMICO: Sano vs Vidrio Esmerilado (GGO) vs Fibrosis Establecida", fontsize=16, y=0.99)
    plt.tight_layout()
    
    img_path = os.path.join(DATA_DIR, "benchmark_modelos_multiclase.png")
    plt.savefig(img_path, dpi=200, bbox_inches='tight')
    print(f"Gráfico de benchmark guardado en: {img_path}", flush=True)
    plt.close()


if __name__ == "__main__":
    ejecutar_benchmark()
