# Informe de Detección de Anomalías Financieras

**Fecha de ejecución:** 2026-09-08 13:20:05  
**Tiempo de ejecución:** 20.26 segundos

## Descripción del Dataset

- **Número de muestras:** 2480
- **Número de características:** 36
- **Características utilizadas:** Close, High, Low, Open, Volume, Volatility_10d, Volatility_20d, Volatility_30d, SMA_10, SMA_30...

## Modelos Evaluados

Se evaluaron 4 modelos de detección de anomalías:

1. **Statistical Baseline (Z-Score):** Método estadístico paramétrico con Z-score móvil
2. **Isolation Forest:** Algoritmo basado en aislamiento mediante árboles aleatorios  
3. **Local Outlier Factor (LOF):** Detección basada en densidad local
4. **LSTM Autoencoder:** Red neuronal recurrente para reconstrucción de secuencias

## Resultados

| Modelo | Precision | Recall | F1-Score | AUC-ROC |
|--------|-----------|--------|----------|---------|
| Statistical Baseline | 0.0270 | 0.1250 | 0.0444 | N/A |
| Isolation Forest | 0.1023 | 0.5625 | 0.1731 | 0.7681 |
| LOF | 0.0000 | 0.0000 | 0.0000 | 0.5757 |
| LSTM Autoencoder | 0.0000 | 0.0000 | 0.0000 | 0.2300 |

## Conclusiones

El modelo con mejor F1-Score fue **Isolation Forest** con 0.1731.

### Recomendaciones

- Baseline: Bueno como referencia pero limitado a distribuciones normales
- Isolation Forest: Robustos y eficientes, buen punto de partida
- LOF: Sensibles a dimensión alta, requiere tuning de vecinos
- LSTM: Capturan patrones temporales pero requieren más datos y computación
