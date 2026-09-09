import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import scipy.stats as stats

# Configuración visual para entorno académico
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)

def run_eda(raw_data_path="data/raw/sp500_historical.csv"):
    print("Iniciando Análisis Exploratorio de Datos (EDA)...")
    os.makedirs("results/eda", exist_ok=True)
    
    # 1. Cargar datos
    df = pd.read_csv(raw_data_path, index_col=0, parse_dates=True, skiprows=[1, 2])
    df.columns = df.columns.str.strip().str.replace('^', '', regex=False)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna()
    
    # Calcular retornos diarios para el análisis
    df['Returns'] = df['Close'].pct_change()
    df = df.dropna()

    # GRÁFICA 1: Evolución del precio y Volumen (Contexto Histórico)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
    ax1.plot(df.index, df['Close'], color='navy', linewidth=1.5)
    ax1.set_title('Evolución Histórica del Índice S&P 500 (2016-2026)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Precio de Cierre (USD)')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    ax2.bar(df.index, df['Volume'], color='gray', alpha=0.5)
    ax2.set_ylabel('Volumen')
    ax2.set_xlabel('Fecha')
    ax2.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('results/eda/01_historia_precio_volumen.png', dpi=300)
    print("Gráfica 1 generada.")

    # GRÁFICA 2: Demostración de Colas Pesadas (Fat Tails)
    plt.figure(figsize=(10, 6))
    returns_clean = df['Returns'] * 100 # En porcentaje
    sns.histplot(returns_clean, bins=100, stat='density', color='skyblue', label='Retornos Empíricos')
    
    # Ajustar campana de Gauss
    mu, std = stats.norm.fit(returns_clean)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2, label=f'Distribución Normal\n($\mu$={mu:.2f}, $\sigma$={std:.2f})')
    
    plt.title('Distribución de Retornos Diarios vs. Distribución Normal (Gaussiana)', fontsize=14, fontweight='bold')
    plt.xlabel('Retorno Diario (%)')
    plt.ylabel('Densidad')
    plt.xlim(-5, 5) # Recortar para ver mejor el centro y colas
    plt.legend()
    plt.savefig('results/eda/02_colas_pesadas.png', dpi=300)
    print("Gráfica 2 generada.")

    # GRÁFICA 3: Agrupamiento de Volatilidad (Volatility Clustering)
    plt.figure(figsize=(12, 5))
    plt.plot(df.index, df['Returns'] * 100, color='crimson', linewidth=0.8)
    plt.title('Agrupamiento de Volatilidad (Volatility Clustering) en S&P 500', fontsize=14, fontweight='bold')
    plt.ylabel('Retorno Diario (%)')
    plt.xlabel('Fecha')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axhline(0, color='black', linewidth=1)
    plt.tight_layout()
    plt.savefig('results/eda/03_volatility_clustering.png', dpi=300)
    print("Gráfica 3 generada.")

    # GRÁFICA 4: Matriz de Correlación (Justifica fallo de LOF por dimensionalidad)
    plt.figure(figsize=(10, 8))
    # Seleccionamos variables clave para no hacer la matriz ilegible
    cols_corr = ['Open', 'High', 'Low', 'Close', 'Volume']
    corr_matrix = df[cols_corr].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.3f', square=True)
    plt.title('Matriz de Correlación de Variables OHLCV', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/eda/04_matriz_correlacion.png', dpi=300)
    print("Gráfica 4 generada.")
    
    print("EDA finalizado con éxito. Revisa la carpeta results/eda/")

if __name__ == "__main__":
    run_eda()