# TFE: Detección de Anomalías Financieras

Este repositorio contiene el código fuente y el flujo de procesamiento (*pipeline*) experimental correspondiente al Trabajo Fin de Estudios: "Sistema Inteligente para la Detección de Anomalías en Carteras Financieras mediante Aprendizaje No Supervisado".

Los mercados financieros son entornos altamente dinámicos caracterizados por el agrupamiento de volatilidad y la presencia de eventos extremos (colas pesadas), como los *Flash Crashes*. Los modelos estadísticos de riesgo tradicionales, que a menudo asumen distribuciones normales en los datos, fracasan a la hora de predecir o identificar estas anomalías estructurales complejas.

Para abordar este desafío, este proyecto desarrolla un sistema automatizado de alertas tempranas utilizando Inteligencia Artificial No Supervisada. Mediante el análisis de grandes volúmenes de series temporales financieras históricas (ingeridas a través de la API de `yfinance`), el sistema aprende de forma autónoma el comportamiento "normal" de las carteras y alerta sobre picos de riesgo significativos.

Siguiendo la metodología CRISP-DM, la arquitectura técnica presenta un análisis comparativo riguroso a través de tres enfoques:

1. Línea Base Estadística: Métodos paramétricos tradicionales (ej. Z-Score móvil) para establecer un umbral mínimo de rendimiento.
2. Machine Learning Clásico: *Isolation Forest* y *Local Outlier Factor* (LOF) para la detección de anomalías espaciales y basadas en densidad local.
3. Deep Learning: *LSTM-Autoencoders* diseñados para capturar dependencias temporales y detectar caídas abruptas a través de umbrales de error de reconstrucción.

## ✨ Características Principales y Aportaciones Técnicas

*   **Validación Histórica (Ground Truth):** A diferencia de las evaluaciones no supervisadas estándar que dependen de ruido sintético, este *pipeline* evalúa los modelos frente a crisis sistémicas históricas reales (ej. la corrección de 2018, el colapso por COVID-19 en 2020, la crisis inflacionaria de 2022 y el *Flash Crash* de agosto de 2024).
*   **Ingeniería de Características de Alta Dimensionalidad:** Generación automatizada de un espacio de características de 36 dimensiones, incluyendo retornos logarítmicos, volatilidades móviles, MACD, RSI, Bandas de Bollinger y retardos temporales.
*   **Inteligencia Artificial Explicable (XAI):** Integración de *TreeSHAP* (SHapley Additive exPlanations) para superar el problema de la "caja negra", transformando las puntuaciones matemáticas de anomalía en alertas semánticas comprensibles para analistas financieros.

## 📊 Síntesis de Resultados

La evaluación empírica demostró que los algoritmos de aislamiento espacial (**Isolation Forest**) superan significativamente a la estadística tradicional y a los enfoques basados en densidad en conjuntos de datos financieros altamente colineales, logrando el mejor equilibrio predictivo (F1-Score: 0.1731, AUC-ROC: 0.7681). Las arquitecturas de aprendizaje profundo (LSTM) mostraron limitaciones debido a la inestabilidad de los gradientes al aplicarse sobre datos diarios OHLCV.

---

## ⚙️ Manual de Usuario e Instrucciones de Instalación

### Requisitos Previos
*   Python 3.8 o superior.
*   Git

### Configuración del Entorno

1. **Clonar el repositorio:**
```bash
git clone https://github.com/Pruizma/TFE-Financial-Anomalies-Detection.git
cd TFE-Fnancial-Anomalies-Detection
```

2. **Crear y activar un entorno virtual (Recomendado):**
```bash
python -m venv venv
# En Windows:
venv\Scripts\activate
# En macOS/Linux:
source venv/bin/activate
```

3. **Instalar las dependencias requeridas:**
```bash
pip install -r requirements.txt
```

### Ejecución del Sistema

El sistema está diseñado para ejecutarse de principio a fin (*end-to-end*) mediante un único script orquestador. Este script ejecutará la ingesta de datos, el preprocesamiento, el entrenamiento de los modelos y la evaluación final:

```bash
python src/04_run_pipeline.py
```

*Nota: El script mostrará las métricas de evaluación a través de la consola y generará gráficos comparativos (`.png`) así como un informe final (`report.md`) dentro del directorio `results/`.*

---
**Autores:** José Manuel Ruiz-Mateos & Daniel Rueda
