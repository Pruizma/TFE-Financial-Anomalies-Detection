"""
Script Principal de Ejecución del Pipeline
==========================================
Ejecuta el flujo completo del TFE:
1. Carga de datos preprocesados
2. Entrenamiento de los 4 modelos
3. Evaluación y comparación
4. Guardado de resultados

Autor: Daniel Rueda
Fecha: 2026
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
from datetime import datetime

# Importar módulos del proyecto
import importlib.util

# Cargar módulos de manera dinámica si están en el path
def load_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Intentar importar localmente
try:
    from anomaly_detection_models import (
        StatisticalBaseline, AnomalyDetectorIF, AnomalyDetectorLOF,
        LSTMAutoencoder, evaluate_model, compare_models
    )
except ImportError:
    print("[INFO] Cargando módulo desde ruta relativa...")
    models_module = load_module_from_path('anomaly_detection_models', 
                                         os.path.join(os.path.dirname(__file__), 
                                                    '03_anomaly_detection_models.py'))
    StatisticalBaseline = models_module.StatisticalBaseline
    AnomalyDetectorIF = models_module.AnomalyDetectorIF
    AnomalyDetectorLOF = models_module.AnomalyDetectorLOF
    LSTMAutoencoder = models_module.LSTMAutoencoder
    evaluate_model = models_module.evaluate_model
    compare_models = models_module.compare_models


def load_processed_data(data_dir='data/processed'):
    """
    Carga los datos procesados del módulo 02.
    
    Args:
        data_dir: Directorio con datos procesados
        
    Returns:
        Diccionario con X_scaled, X_seq, feature_names, df
    """
    print("=" * 70)
    print("CARGA DE DATOS PROCESADOS")
    print("=" * 70)
    
    # Buscar archivos más recientes
    import glob
    
    # Buscar archivos de datos escalados
    csv_files = glob.glob(os.path.join(data_dir, 'X_scaled_*.npy'))
    
    if not os.path.exists(data_dir):
        print(f"[ERROR] Directorio no encontrado: {data_dir}")
        print("[INFO] Ejecuta primero: 02_data_preprocessing.py")
        return None
    
    # Verificar qué archivos existen
    files_in_dir = os.listdir(data_dir)
    print(f"\n[INFO] Archivos en {data_dir}:")
    for f in sorted(files_in_dir):
        print(f"  - {f}")
    
    # Cargar datos
    data = {}
    
    # Intentar cargar X_scaled
    X_scaled_path = os.path.join(data_dir, 'X_scaled.npy')
    if os.path.exists(X_scaled_path):
        data['X'] = np.load(X_scaled_path)
        print(f"\n[OK] X escalado cargado: {data['X'].shape}")
    else:
        # Buscar archivo con timestamp
        csv_files = [f for f in files_in_dir if f.startswith('X_scaled_') and f.endswith('.npy')]
        if csv_files:
            most_recent = sorted(csv_files)[-1]
            data['X'] = np.load(os.path.join(data_dir, most_recent))
            print(f"\n[OK] X escalado cargado ({most_recent}): {data['X'].shape}")
    
    # Cargar secuencias para LSTM
    X_seq_path = os.path.join(data_dir, 'X_sequences.npy')
    if os.path.exists(X_seq_path):
        data['X_seq'] = np.load(X_seq_path)
        print(f"[OK] Secuencias cargadas: {data['X_seq'].shape}")
    else:
        seq_files = [f for f in files_in_dir if 'sequences' in f and f.endswith('.npy')]
        if seq_files:
            most_recent = sorted(seq_files)[-1]
            data['X_seq'] = np.load(os.path.join(data_dir, most_recent))
            print(f"[OK] Secuencias cargadas ({most_recent}): {data['X_seq'].shape}")
        else:
            print("[WARNING] No se encontraron secuencias para LSTM")
            data['X_seq'] = None
    
    # Cargar nombres de features
    features_path = os.path.join(data_dir, 'feature_names.csv')
    if os.path.exists(features_path):
        df_features = pd.read_csv(features_path)
        data['feature_names'] = df_features['features'].tolist()
        print(f"[OK] {len(data['feature_names'])} feature names cargados")
    
    # Cargar DataFrame original (último archivo features)
    csv_feature_files = [f for f in files_in_dir if f.startswith('features_df_') and f.endswith('.csv')]
    if csv_feature_files:
        most_recent = sorted(csv_feature_files)[-1]
        data['df'] = pd.read_csv(os.path.join(data_dir, most_recent), index_col=0, parse_dates=True)
        print(f"[OK] DataFrame cargado ({most_recent}): {data['df'].shape}")
    
    return data


def create_synthetic_anomalies(X, contamination=0.05):
    """
    Crea anomalías sintéticas si no hay etiquetas reales disponibles.
    
    Args:
        X: Datos de entrada
        contamination: Porcentaje de anomalías a crear
        
    Returns:
        y_true: Etiquetas sintéticas (1: normal, -1: anomalía)
    """
    np.random.seed(42)
    n_samples = len(X)
    n_anomalies = int(n_samples * contamination)
    
    # Seleccionar índices aleatorios como anomalías
    anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
    y_true = np.ones(n_samples)
    y_true[anomaly_indices] = -1
    
    print(f"\n[INFO] Anomalías sintéticas creadas: {n_anomalies} ({contamination*100:.1f}%)")
    
    return y_true.astype(int)


def split_train_test(X, y, test_size=0.2):
    """
    Divide los datos en entrenamiento y test manteniendo temporalidad.
    
    Args:
        X: Features
        y: Etiquetas
        test_size: Proporción de test
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    split_idx = int(len(X) * (1 - test_size))
    
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"[INFO] División temporal: Train={len(X_train)}, Test={len(X_test)}")
    
    return X_train, X_test, y_train, y_test


def run_baseline_model(X_train, X_test, y_train, y_test):
    """Ejecuta el modelo Baseline estadístico."""
    print("\n" + "=" * 70)
    print("1. STATISTICAL BASELINE (Z-Score)")
    print("=" * 70)
    
    model = StatisticalBaseline(window=20, threshold=3.0)
    model.fit(X_train)
    y_pred = model.predict(X_test)
    scores = model.decision_function(X_test)
    
    results = evaluate_model(y_test, y_pred, scores, "Statistical Baseline")
    
    return results


def run_isolation_forest(X_train, X_test, y_train, y_test):
    """Ejecuta el modelo Isolation Forest."""
    print("\n" + "=" * 70)
    print("2. ISOLATION FOREST")
    print("=" * 70)
    
    model = AnomalyDetectorIF(contamination=0.05, n_estimators=100)
    model.fit(X_train)
    y_pred = model.predict(X_test)
    scores = model.decision_function(X_test)
    
    results = evaluate_model(y_test, y_pred, scores, "Isolation Forest")
    
    # Guardar modelo
    os.makedirs('models', exist_ok=True)
    model.save('models/isolation_forest.pkl')
    print("[OK] Modelo guardado en: models/isolation_forest.pkl")
    
    return results


def run_lof(X_train, X_test, y_train, y_test):
    """Ejecuta el modelo Local Outlier Factor."""
    print("\n" + "=" * 70)
    print("3. LOCAL OUTLIER FACTOR (LOF)")
    print("=" * 70)
    
    # LOF no tiene predict separado, hay que reentrenar con todos los datos
    X_combined = np.vstack([X_train, X_test])
    y_combined = np.concatenate([y_train, y_test])
    
    model = AnomalyDetectorLOF(n_neighbors=20, contamination=0.05)
    y_pred_combined = model.fit_predict(X_combined)
    
    # Extraer predicciones para test
    y_pred = y_pred_combined[len(X_train):]
    
    # Calcular scores (reentrenar con novelty=True)
    scores = model.decision_function(X_test)
    
    results = evaluate_model(y_test, y_pred, scores, "LOF")
    
    return results


def run_lstm_autoencoder(X_train_seq, X_test_seq, y_train, y_test):
    """Ejecuta el modelo LSTM Autoencoder."""
    print("\n" + "=" * 70)
    print("4. LSTM AUTOENCODER")
    print("=" * 70)
    
    if X_train_seq is None or X_test_seq is None:
        print("[WARNING] No hay secuencias disponibles para LSTM. Saltando...")
        return None
    
    timesteps = X_train_seq.shape[1]
    n_features = X_train_seq.shape[2]
    
    # Dividir training para validación
    split_val = int(len(X_train_seq) * 0.8)
    X_train_split = X_train_seq[:split_val]
    X_val_split = X_train_seq[split_val:]
    
    model = LSTMAutoencoder(
        timesteps=timesteps,
        n_features=n_features,
        lstm_units=[64, 32],
        learning_rate=0.001,
        threshold_percentile=95
    )
    
    model.fit(X_train_split, X_val_split, epochs=50, batch_size=32)
    y_pred = model.predict(X_test_seq)
    scores = model.decision_function(X_test_seq)
    
    # Ajustar y_test a la longitud de las secuencias
    y_test_adj = y_test[timesteps-1:timesteps-1+len(y_pred)]
    
    results = evaluate_model(y_test_adj, y_pred, scores, "LSTM Autoencoder")
    
    # Guardar modelo
    model.save('models/lstm_autoencoder.keras')
    print("[OK] Modelo guardado en: models/lstm_autoencoder.keras")
    
    return results


def save_results(all_results, output_dir='results'):
    """
    Guarda los resultados de la evaluación.
    
    Args:
        all_results: Lista de diccionarios con resultados
        output_dir: Directorio de salida
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Crear DataFrame
    results_df = pd.DataFrame([{
        'model': r['model'],
        'precision': r['precision'],
        'recall': r['recall'],
        'f1': r['f1'],
        'auc': r['auc']
    } for r in all_results])
    
    # Guardar CSV
    csv_path = os.path.join(output_dir, f'results_{timestamp}.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\n[OK] Resultados guardados en: {csv_path}")
    
    # Mostrar tabla
    print("\n" + "=" * 70)
    print("COMPARACIÓN FINAL DE MODELOS")
    print("=" * 70)
    print(results_df.to_string(index=False))
    
    return results_df


def generate_markdown_report(data, all_results, execution_time, output_path='results/report.md'):
    """
    Genera un informe en Markdown con los resultados.
    
    Args:
        data: Diccionario con datos cargados
        all_results: Lista de resultados
        execution_time: Tiempo de ejecución
        output_path: Ruta del informe
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    report = f"""# Informe de Detección de Anomalías Financieras

**Fecha de ejecución:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Tiempo de ejecución:** {execution_time:.2f} segundos

## Descripción del Dataset

- **Número de muestras:** {len(data.get('X', []))}
- **Número de características:** {len(data.get('feature_names', []))}
- **Características utilizadas:** {', '.join(data.get('feature_names', [])[:10])}...

## Modelos Evaluados

Se evaluaron 4 modelos de detección de anomalías:

1. **Statistical Baseline (Z-Score):** Método estadístico paramétrico con Z-score móvil
2. **Isolation Forest:** Algoritmo basado en aislamiento mediante árboles aleatorios  
3. **Local Outlier Factor (LOF):** Detección basada en densidad local
4. **LSTM Autoencoder:** Red neuronal recurrente para reconstrucción de secuencias

## Resultados

"""
    
    # Añadir tabla de resultados
    report += "| Modelo | Precision | Recall | F1-Score | AUC-ROC |\n"
    report += "|--------|-----------|--------|----------|---------|\n"
    
    for r in all_results:
        auc = f"{r['auc']:.4f}" if r['auc'] is not None else "N/A"
        report += f"| {r['model']} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} | {auc} |\n"
    
    report += "\n## Conclusiones\n\n"
    
    # Encontrar mejor modelo por F1
    best_model = max(all_results, key=lambda x: x['f1'])
    report += f"El modelo con mejor F1-Score fue **{best_model['model']}** con {best_model['f1']:.4f}.\n\n"
    
    report += "### Recomendaciones\n\n"
    report += "- Baseline: Bueno como referencia pero limitado a distribuciones normales\n"
    report += "- Isolation Forest: Robustos y eficientes, buen punto de partida\n"
    report += "- LOF: Sensibles a dimensión alta, requiere tuning de vecinos\n"
    report += "- LSTM: Capturan patrones temporales pero requieren más datos y computación\n"
    
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"\n[OK] Informe Markdown guardado en: {output_path}")


def main():
    """Función principal que ejecuta todo el pipeline."""
    import time
    start_time = time.time()
    
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETO DE DETECCIÓN DE ANOMALÍAS")
    print("=" * 70)
    
    # 1. Cargar datos
    data = load_processed_data()
    if data is None:
        print("[ERROR] No se pudieron cargar los datos")
        sys.exit(1)
    
    X = data['X']
    X_seq = data.get('X_seq')
    
    # 2. Crear etiquetas sintéticas (o cargar si existen reales)
    y = create_synthetic_anomalies(X, contamination=0.05)
    
    # 3. Dividir train/test temporalmente
    X_train, X_test, y_train, y_test = split_train_test(X, y, test_size=0.2)
    
    # Para LSTM, dividir secuencias también
    X_train_seq, X_test_seq = None, None
    if X_seq is not None:
        split_idx = int(len(X_seq) * 0.8)
        X_train_seq = X_seq[:split_idx]
        X_test_seq = X_seq[split_idx:]
    
    # 4. Ejecutar modelos
    all_results = []
    
    # Baseline
    result = run_baseline_model(X_train, X_test, y_train, y_test)
    if result:
        all_results.append(result)
    
    # Isolation Forest
    result = run_isolation_forest(X_train, X_test, y_train, y_test)
    if result:
        all_results.append(result)
    
    # LOF
    result = run_lof(X_train, X_test, y_train, y_test)
    if result:
        all_results.append(result)
    
    # LSTM (si hay secuencias)
    if X_train_seq is not None and X_test_seq is not None:
        result = run_lstm_autoencoder(X_train_seq, X_test_seq, y_train, y_test)
        if result:
            all_results.append(result)
    
    # 5. Guardar y comparar resultados
    results_df = save_results(all_results)
    
    # 6. Generar gráficas comparativas
    try:
        compare_models(results_df, output_path='results/model_comparison.png')
    except Exception as e:
        print(f"[WARNING] No se pudo generar gráfica: {e}")
    
    # 7. Generar informe
    execution_time = time.time() - start_time
    generate_markdown_report(data, all_results, execution_time)
    
    print("\n" + "=" * 70)
    print(f"PIPELINE COMPLETADO en {execution_time:.2f} segundos")
    print("=" * 70)
    print("\nArchivos generados:")
    print("  - models/isolation_forest.pkl")
    print("  - models/lstm_autoencoder.keras")
    print("  - results/results_*.csv")
    print("  - results/model_comparison.png")
    print("  - results/report.md")


if __name__ == "__main__":
    # Verificar argumentos
    import argparse
    parser = argparse.ArgumentParser(description='Pipeline de Detección de Anomalías')
    parser.add_argument('--data', type=str, default='data/processed',
                       help='Directorio con datos procesados')
    parser.add_argument('--skip-lstm', action='store_true',
                       help='Omitir entrenamiento de LSTM (más rápido)')
    args = parser.parse_args()
    
    print("Ejecutando pipeline... Usa --help para opciones")
    print(f"Directorio de datos: {args.data}")
    
    main()
