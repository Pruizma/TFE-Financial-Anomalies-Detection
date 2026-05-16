"""
Módulo de Preprocesamiento de Datos Financieros
===============================================
Prepara los datos descargados para el análisis de anomalías.
Incluye limpieza, normalización, feature engineering y preparación de secuencias.

Autor: Daniel Rueda
Fecha: 2026
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import pickle
import os
from datetime import datetime


def load_raw_data(file_path):
    """
    Carga los datos financieros crudos descargados.
    
    Args:
        file_path: Ruta al archivo CSV con los datos crudos
        
    Returns:
        DataFrame con los datos cargados
    """
    print(f"[INFO] Cargando datos desde: {file_path}")
    
    try:
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        print(f"[INFO] Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
        print(f"[INFO] Rango de fechas: {df.index[0]} a {df.index[-1]}")
        return df
    except Exception as e:
        print(f"[ERROR] No se pudieron cargar los datos: {e}")
        return None


def clean_data(df):
    """
    Limpia los datos: elimina valores nulos, duplicados y outliers extremos.
    
    Args:
        df: DataFrame con datos crudos
        
    Returns:
        DataFrame limpio
    """
    print("\n[INFO] Limpiando datos...")
    initial_rows = len(df)
    
    # Eliminar valores nulos
    df_clean = df.dropna()
    null_removed = initial_rows - len(df_clean)
    
    # Eliminar duplicados
    df_clean = df_clean.drop_duplicates()
    dup_removed = initial_rows - null_removed - len(df_clean)
    
    # Manejar outliers extremos (z-score > 5)
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
        df_clean = df_clean[z_scores < 5]
    
    outliers_removed = initial_rows - null_removed - dup_removed - len(df_clean)
    
    print(f"  - Valores nulos eliminados: {null_removed}")
    print(f"  - Duplicados eliminados: {dup_removed}")
    print(f"  - Outliers extremos eliminados: {outliers_removed}")
    print(f"  - Filas finales: {len(df_clean)}")
    
    return df_clean


def calculate_returns(df, price_col='Close'):
    """
    Calcula retornos logarítmicos y simples.
    
    Args:
        df: DataFrame con datos de precios
        price_col: Nombre de la columna de precios
        
    Returns:
        DataFrame con retornos añadidos
    """
    print("\n[INFO] Calculando retornos...")
    
    # Retornos simples
    df['Returns_Simple'] = df[price_col].pct_change()
    
    # Retornos logarítmicos (más estables para análisis estadístico)
    df['Returns_Log'] = np.log(df[price_col] / df[price_col].shift(1))
    
    # Retornos acumulados
    df['Returns_Cumulative'] = (1 + df['Returns_Simple']).cumprod() - 1
    
    print("  - Retornos simples: ✓")
    print("  - Retornos logarítmicos: ✓")
    print("  - Retornos acumulados: ✓")
    
    return df


def calculate_volatility(df, windows=[10, 20, 30]):
    """
    Calcula volatilidad móvil en diferentes ventanas temporales.
    
    Args:
        df: DataFrame con retornos
        windows: Lista de ventanas para volatilidad
        
    Returns:
        DataFrame con volatilidades añadidas
    """
    print("\n[INFO] Calculando volatilidad...")
    
    for window in windows:
        # Volatilidad anualizada
        df[f'Volatility_{window}d'] = df['Returns_Log'].rolling(window=window).std() * np.sqrt(252)
    
    print(f"  - Volatilidad calculada para ventanas: {windows}")
    return df


def calculate_technical_indicators(df, price_col='Close', high_col='High', low_col='Low'):
    """
    Calcula indicadores técnicos útiles para detección de anomalías.
    
    Args:
        df: DataFrame con datos OHLC
        price_col, high_col, low_col: Nombres de columnas
        
    Returns:
        DataFrame con indicadores técnicos
    """
    print("\n[INFO] Calculando indicadores técnicos...")
    
    # Medias móviles
    df['SMA_10'] = df[price_col].rolling(window=10).mean()
    df['SMA_30'] = df[price_col].rolling(window=30).mean()
    df['EMA_10'] = df[price_col].ewm(span=10).mean()
    
    # RSI (Relative Strength Index)
    delta = df[price_col].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df[price_col].ewm(span=12).mean()
    exp2 = df[price_col].ewm(span=26).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    
    # Bollinger Bands
    df['BB_Middle'] = df[price_col].rolling(window=20).mean()
    bb_std = df[price_col].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
    df['BB_Position'] = (df[price_col] - df['BB_Lower']) / df['BB_Width']
    
    # ATR (Average True Range) - medida de volatilidad
    high_low = df[high_col] - df[low_col]
    high_close = np.abs(df[high_col] - df[price_col].shift())
    low_close = np.abs(df[low_col] - df[price_col].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(window=14).mean()
    
    print("  - Medias móviles: ✓")
    print("  - RSI: ✓")
    print("  - MACD: ✓")
    print("  - Bandas de Bollinger: ✓")
    print("  - ATR: ✓")
    
    return df


def create_lag_features(df, features, lags=[1, 2, 3, 5, 10]):
    """
    Crea características retardadas (lag features) para capturar dependencias temporales.
    
    Args:
        df: DataFrame con características
        features: Lista de columnas para crear lags
        lags: Lista de retardos a crear
        
    Returns:
        DataFrame con características retardadas
    """
    print("\n[INFO] Creando características retardadas...")
    
    for feature in features:
        for lag in lags:
            df[f'{feature}_Lag_{lag}'] = df[feature].shift(lag)
    
    total_features = len(features) * len(lags)
    print(f"  - {total_features} características de lag creadas")
    
    return df


def prepare_features_for_ml(df, feature_cols=None, target_col=None):
    """
    Prepara las características finales para los modelos de ML.
    
    Args:
        df: DataFrame con todas las características
        feature_cols: Lista de columnas a usar (si None, usa todas las numéricas)
        target_col: Columna objetivo (para supervisado)
        
    Returns:
        X: Features preparados
        y: Target (si se especifica)
        feature_names: Nombres de características
    """
    print("\n[INFO] Preparando features para ML...")
    
    # Seleccionar columnas numéricas si no se especifican
    if feature_cols is None:
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Excluir columnas no deseadas
        exclude = ['Returns_Simple', 'Returns_Log', 'Returns_Cumulative']
        feature_cols = [c for c in feature_cols if c not in exclude]
    
    # Limpiar NaN
    df_clean = df[feature_cols].dropna()
    
    X = df_clean.values
    feature_names = feature_cols
    
    print(f"  - Características seleccionadas: {len(feature_cols)}")
    print(f"  - Muestras válidas: {len(df_clean)}")
    
    if target_col:
        y = df.loc[df_clean.index, target_col].values
        return X, y, feature_names
    
    return X, feature_names


def scale_features(X, method='standard', save_scaler=True, scaler_path='models/scaler.pkl'):
    """
    Escala las características para modelos de ML.
    
    Args:
        X: Matriz de características
        method: 'standard' o 'minmax'
        save_scaler: Si guardar el scaler entrenado
        scaler_path: Ruta para guardar el scaler
        
    Returns:
        X_scaled: Características escaladas
        scaler: Objeto scaler entrenado
    """
    print(f"\n[INFO] Escalando características (método: {method})...")
    
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("Método debe ser 'standard' o 'minmax'")
    
    X_scaled = scaler.fit_transform(X)
    
    if save_scaler:
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"  - Scaler guardado en: {scaler_path}")
    
    print(f"  - Media original: {X.mean(axis=0)[:3].round(3)}...")
    print(f"  - Media escalada: {X_scaled.mean(axis=0)[:3].round(3)}...")
    
    return X_scaled, scaler


def create_sequences(X, y=None, timesteps=30):
    """
    Crea secuencias temporales para modelos LSTM/Autoencoder.
    
    Args:
        X: Características escaladas
        y: Targets opcionales
        timesteps: Longitud de la secuencia temporal
        
    Returns:
        X_seq: Secuencias de características
        y_seq: Targets correspondientes (si se proporcionan)
    """
    print(f"\n[INFO] Creando secuencias temporales (timesteps={timesteps})...")
    
    X_seq = []
    y_seq = []
    
    for i in range(len(X) - timesteps + 1):
        X_seq.append(X[i:i + timesteps])
        if y is not None:
            y_seq.append(y[i + timesteps - 1])
    
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq) if y is not None else None
    
    print(f"  - Secuencias creadas: {len(X_seq)}")
    print(f"  - Forma de X_seq: {X_seq.shape}")
    
    if y is not None:
        return X_seq, y_seq
    return X_seq


def save_processed_data(data_dict, output_dir='data/processed'):
    """
    Guarda los datos procesados para su uso posterior.
    
    Args:
        data_dict: Diccionario con datos a guardar
        output_dir: Directorio de salida
    """
    print(f"\n[INFO] Guardando datos procesados en: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for name, data in data_dict.items():
        if isinstance(data, pd.DataFrame):
            filepath = os.path.join(output_dir, f"{name}_{timestamp}.csv")
            data.to_csv(filepath)
            print(f"  - {name}.csv guardado")
        elif isinstance(data, np.ndarray):
            filepath = os.path.join(output_dir, f"{name}_{timestamp}.npy")
            np.save(filepath, data)
            print(f"  - {name}.npy guardado")
    
    return timestamp


def run_full_preprocessing(input_path, output_dir='data/processed', 
                          create_sequences_flag=True, timesteps=30):
    """
    Ejecuta el pipeline completo de preprocesamiento.
    
    Args:
        input_path: Ruta al archivo CSV de datos crudos
        output_dir: Directorio para datos procesados
        create_sequences_flag: Si crear secuencias para LSTM
        timesteps: Longitud de secuencias temporales
        
    Returns:
        Diccionario con todos los datos procesados
    """
    print("=" * 60)
    print("PIPELINE DE PREPROCESAMIENTO DE DATOS FINANCIEROS")
    print("=" * 60)
    
    # 1. Cargar datos
    df = load_raw_data(input_path)
    if df is None:
        return None
    
    # 2. Limpiar datos
    df = clean_data(df)
    
    # 3. Calcular retornos
    df = calculate_returns(df)
    
    # 4. Calcular volatilidad
    df = calculate_volatility(df)
    
    # 5. Indicadores técnicos
    if all(col in df.columns for col in ['High', 'Low', 'Close']):
        df = calculate_technical_indicators(df)
    
    # 6. Características retardadas
    lag_features = ['Returns_Log', 'Volatility_20d', 'RSI']
    available_lag_features = [f for f in lag_features if f in df.columns]
    if available_lag_features:
        df = create_lag_features(df, available_lag_features)
    
    # 7. Preparar features para ML
    X, feature_names = prepare_features_for_ml(df)
    
    # 8. Escalar features
    X_scaled, scaler = scale_features(X)
    
    # 9. Crear secuencias (para LSTM)
    if create_sequences_flag:
        X_seq = create_sequences(X_scaled, timesteps=timesteps)
    else:
        X_seq = None
    
    # 10. Guardar resultados
    data_to_save = {
        'features_df': df,
        'X_scaled': X_scaled,
        'feature_names': pd.DataFrame({'features': feature_names})
    }
    
    if X_seq is not None:
        data_to_save['X_sequences'] = X_seq
    
    timestamp = save_processed_data(data_to_save, output_dir)
    
    print("\n" + "=" * 60)
    print("PREPROCESAMIENTO COMPLETADO")
    print("=" * 60)
    
    return {
        'df': df,
        'X_scaled': X_scaled,
        'X_seq': X_seq,
        'scaler': scaler,
        'feature_names': feature_names,
        'timestamp': timestamp
    }


if __name__ == "__main__":
    # Ejemplo de uso
    INPUT_FILE = "data/raw/sp500_historical.csv"
    OUTPUT_DIR = "data/processed"
    
    if os.path.exists(INPUT_FILE):
        results = run_full_preprocessing(INPUT_FILE, OUTPUT_DIR)
        
        if results:
            print("\n[RESUMEN]")
            print(f"- Datos procesados: {len(results['df'])} filas")
            print(f"- Características: {len(results['feature_names'])}")
            print(f"- Forma de X_scaled: {results['X_scaled'].shape}")
            if results['X_seq'] is not None:
                print(f"- Forma de secuencias: {results['X_seq'].shape}")
    else:
        print(f"[ERROR] No se encontró el archivo: {INPUT_FILE}")
        print("[INFO] Ejecuta primero: 01_data_ingestion.py")
