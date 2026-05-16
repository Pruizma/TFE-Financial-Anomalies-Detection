"""
Módulo de Modelos de Detección de Anomalías
===========================================
Implementa 4 enfoques para detección de anomalías en series temporales financieras:
1. Baseline estadístico (Z-Score móvil)
2. Isolation Forest
3. Local Outlier Factor (LOF)
4. LSTM Autoencoder

Autor: Daniel Rueda
Fecha: 2026
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix
import pickle
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Para el LSTM
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, TimeDistributed, RepeatVector
from tensorflow.keras.callbacks import EarlyStopping


class StatisticalBaseline:
    """
    Modelo Baseline: Detección de anomalías mediante Z-Score móvil.
    Metodología estadística paramétrica tradicional.
    """
    
    def __init__(self, window=20, threshold=3.0):
        self.window = window
        self.threshold = threshold
        self.mean = None
        self.std = None
        
    def fit(self, X):
        """Calcula estadísticas móviles."""
        df = pd.DataFrame(X)
        self.mean = df.rolling(window=self.window, min_periods=1).mean()
        self.std = df.rolling(window=self.window, min_periods=1).std()
        return self
        
    def predict(self, X):
        """
        Detecta anomalías basándose en Z-Score.
        Retorna: 1 para normal, -1 para anomalía
        """
        df = pd.DataFrame(X)
        z_score = np.abs((df - self.mean) / (self.std + 1e-10))
        
        # Una anomalía si cualquier feature excede el umbral
        anomalies = (z_score > self.threshold).any(axis=1).astype(int)
        return np.where(anomalies == 1, -1, 1)
    
    def decision_function(self, X):
        """Retorna scores de anomalía (mayor = más anómalo)."""
        df = pd.DataFrame(X)
        z_score = np.abs((df - self.mean) / (self.std + 1e-10))
        return z_score.max(axis=1).values


class AnomalyDetectorIF:
    """
    Isolation Forest: Detecta anomalías basándose en aislamiento.
    Las anomalías son instancias que se separan fácilmente del resto.
    """
    
    def __init__(self, contamination=0.05, n_estimators=100, random_state=42):
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )
        self.contamination = contamination
        
    def fit(self, X):
        """Entrena el modelo Isolation Forest."""
        print(f"[INFO] Entrenando Isolation Forest (contamination={self.contamination})...")
        self.model.fit(X)
        print("[OK] Isolation Forest entrenado")
        return self
        
    def predict(self, X):
        """Predice: 1 para normal, -1 para anomalía."""
        return self.model.predict(X)
        
    def decision_function(self, X):
        """Retorna scores de anomalía (más negativo = más anómalo)."""
        return self.model.decision_function(X)
        
    def save(self, filepath):
        """Guarda el modelo entrenado."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
            
    def load(self, filepath):
        """Carga un modelo guardado."""
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)


class AnomalyDetectorLOF:
    """
    Local Outlier Factor (LOF): Detecta anomalías basándose en densidad local.
    Compara la densidad local de cada punto con la de sus vecinos.
    """
    
    def __init__(self, n_neighbors=20, contamination=0.05):
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.model = None
        
    def fit_predict(self, X):
        """
        Entrena y predice en un solo paso (LOF no tiene predict separado).
        Retorna: 1 para normal, -1 para anomalía
        """
        print(f"[INFO] Entrenando LOF (n_neighbors={self.n_neighbors})...")
        self.model = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination,
            n_jobs=-1
        )
        predictions = self.model.fit_predict(X)
        print("[OK] LOF entrenado")
        return predictions
        
    def decision_function(self, X):
        """Retorna scores de anomalía (LOF score negativo = más anómalo)."""
        # Para LOF hay que reentrenar para nuevos datos
        lof = LocalOutlierFactor(n_neighbors=self.n_neighbors, novelty=True)
        lof.fit(X)
        return -lof.decision_function(X)


class LSTMAutoencoder:
    """
    LSTM Autoencoder: Detecta anomalías mediante reconstrucción de secuencias.
    Usa redes neuronales recurrentes para capturar dependencias temporales.
    """
    
    def __init__(self, timesteps=30, n_features=None, 
                 lstm_units=[64, 32], learning_rate=0.001,
                 threshold_percentile=95):
        self.timesteps = timesteps
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.learning_rate = learning_rate
        self.threshold_percentile = threshold_percentile
        self.model = None
        self.threshold = None
        self.history = None
        
    def build_model(self):
        """Construye la arquitectura del autoencoder LSTM."""
        print(f"[INFO] Construyendo LSTM Autoencoder...")
        print(f"  - Timesteps: {self.timesteps}")
        print(f"  - Features: {self.n_features}")
        print(f"  - LSTM units: {self.lstm_units}")
        
        # Encoder
        model = Sequential([
            LSTM(self.lstm_units[0], activation='relu', 
                 input_shape=(self.timesteps, self.n_features), return_sequences=True),
            Dropout(0.2),
            LSTM(self.lstm_units[1], activation='relu', return_sequences=False),
            RepeatVector(self.timesteps),
            # Decoder
            LSTM(self.lstm_units[1], activation='relu', return_sequences=True),
            Dropout(0.2),
            LSTM(self.lstm_units[0], activation='relu', return_sequences=True),
            TimeDistributed(Dense(self.n_features))
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        
        self.model = model
        print("[OK] Modelo construido")
        return model
        
    def fit(self, X_train, X_val=None, epochs=50, batch_size=32):
        """Entrena el autoencoder."""
        if self.model is None:
            self.n_features = X_train.shape[2]
            self.build_model()
        
        print(f"[INFO] Entrenando LSTM Autoencoder...")
        
        callbacks = [
            EarlyStopping(monitor='val_loss' if X_val is not None else 'loss',
                         patience=10, restore_best_weights=True)
        ]
        
        validation_data = (X_val, X_val) if X_val is not None else None
        
        self.history = self.model.fit(
            X_train, X_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        # Calcular umbral basado en error de reconstrucción del training
        train_pred = self.model.predict(X_train)
        train_mse = np.mean(np.power(X_train - train_pred, 2), axis=(1, 2))
        self.threshold = np.percentile(train_mse, self.threshold_percentile)
        
        print(f"[OK] Entrenamiento completado. Umbral de error: {self.threshold:.6f}")
        return self
        
    def predict(self, X):
        """
        Predice anomalías basándose en error de reconstrucción.
        Retorna: 1 para normal, -1 para anomalía
        """
        predictions = self.model.predict(X)
        mse = np.mean(np.power(X - predictions, 2), axis=(1, 2))
        
        return np.where(mse > self.threshold, -1, 1)
        
    def decision_function(self, X):
        """Retorna error de reconstrucción (mayor = más anómalo)."""
        predictions = self.model.predict(X)
        mse = np.mean(np.power(X - predictions, 2), axis=(1, 2))
        return mse
        
    def save(self, filepath):
        """Guarda el modelo entrenado."""
        self.model.save(filepath)
        # Guardar también el umbral
        with open(filepath + '_threshold.pkl', 'wb') as f:
            pickle.dump({
                'threshold': self.threshold,
                'timesteps': self.timesteps,
                'n_features': self.n_features
            }, f)
            
    def load(self, filepath):
        """Carga un modelo guardado."""
        self.model = tf.keras.models.load_model(filepath)
        with open(filepath + '_threshold.pkl', 'rb') as f:
            params = pickle.load(f)
            self.threshold = params['threshold']
            self.timesteps = params['timesteps']
            self.n_features = params['n_features']


def evaluate_model(y_true, y_pred, scores, model_name):
    """
    Evalúa el rendimiento de un modelo de detección de anomalías.
    
    Args:
        y_true: Etiquetas reales (1: normal, -1: anomalía)
        y_pred: Predicciones (1: normal, -1: anomalía)
        scores: Scores de anomalía
        model_name: Nombre del modelo
        
    Returns:
        dict con métricas
    """
    print(f"\n{'='*60}")
    print(f"EVALUACIÓN: {model_name}")
    print('='*60)
    
    # Convertir a formato binario (0: normal, 1: anomalía)
    y_true_bin = np.where(y_true == -1, 1, 0)
    y_pred_bin = np.where(y_pred == -1, 1, 0)
    
    # Calcular métricas
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_bin, y_pred_bin, average='binary', zero_division=0
    )
    
    # Matriz de confusión
    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin).ravel()
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"\nMatriz de confusión:")
    print(f"  TN: {tn}, FP: {fp}")
    print(f"  FN: {fn}, TP: {tp}")
    
    # AUC-ROC si hay scores
    try:
        if scores is not None and len(np.unique(y_true_bin)) > 1:
            # Normalizar scores para ROC
            scores_norm = (scores - scores.min()) / (scores.max() - scores.min())
            auc = roc_auc_score(y_true_bin, scores_norm)
            print(f"AUC-ROC: {auc:.4f}")
        else:
            auc = None
            print("AUC-ROC: No calculable (scores no disponibles o solo una clase)")
    except:
        auc = None
        print("AUC-ROC: Error en cálculo")
    
    return {
        'model': model_name,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}
    }


def compare_models(results_df, output_path='results/model_comparison.png'):
    """
    Compara visualmente el rendimiento de los modelos.
    
    Args:
        results_df: DataFrame con resultados de todos los modelos
        output_path: Ruta para guardar la gráfica
    """
    import matplotlib.pyplot as plt
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Precision
    axes[0, 0].bar(results_df['model'], results_df['precision'], color='skyblue')
    axes[0, 0].set_title('Precision por Modelo')
    axes[0, 0].set_ylabel('Precision')
    axes[0, 0].set_ylim(0, 1)
    
    # Recall
    axes[0, 1].bar(results_df['model'], results_df['recall'], color='lightgreen')
    axes[0, 1].set_title('Recall por Modelo')
    axes[0, 1].set_ylabel('Recall')
    axes[0, 1].set_ylim(0, 1)
    
    # F1-Score
    axes[1, 0].bar(results_df['model'], results_df['f1'], color='salmon')
    axes[1, 0].set_title('F1-Score por Modelo')
    axes[1, 0].set_ylabel('F1-Score')
    axes[1, 0].set_ylim(0, 1)
    
    # AUC
    auc_data = results_df.dropna(subset=['auc'])
    if not auc_data.empty:
        axes[1, 1].bar(auc_data['model'], auc_data['auc'], color='gold')
        axes[1, 1].set_title('AUC-ROC por Modelo')
        axes[1, 1].set_ylabel('AUC')
        axes[1, 1].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n[OK] Gráfica de comparación guardada en: {output_path}")
    
    return fig


if __name__ == "__main__":
    print("=" * 70)
    print("MÓDULO DE DETECCIÓN DE ANOMALÍAS - 4 MODELOS")
    print("=" * 70)
    print("\nEste módulo implementa:")
    print("  1. Statistical Baseline (Z-Score)")
    print("  2. Isolation Forest")
    print("  3. Local Outlier Factor (LOF)")
    print("  4. LSTM Autoencoder")
    print("\nImporta este módulo desde otro script para entrenar los modelos.")
    print("=" * 70)
