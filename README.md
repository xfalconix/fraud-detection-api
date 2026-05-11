# Red Neuronal Autoencoder para Deteccion de Fraude en Tarjetas de Credito

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Keras](https://img.shields.io/badge/Keras-2.x-red.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-green.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

## Descripcion

Este proyecto implementa un **Autoencoder** basado en redes neuronales profundas para la deteccion de transacciones fraudulentas en tarjetas de credito. El modelo aprende a reconstruir patrones de transacciones legitimas; aquellas con alto error de reconstruccion se clasifican como anomalias potenciales (fraudes).

**Autor:** Carlos Falconi - Malaga, Espana  
**Fecha:** Diciembre 2025

---

## Enfoque Tecnico

### Arquitectura del Autoencoder

```
Input (29 features)
    |
    v
Dense(30)  --> 900 params
    |
    v
Dense(20)  --> 620 params
    |
    v
Dense(10)  --> 210 params
    |
    v
Dense(5)   --> 55 params   <-- Bottleneck (codigo latent)
    |
    v
Dense(10)  --> 60 params
    |
    v
Dense(20)  --> 220 params
    |
    v
Dense(30)  --> 630 params
    |
    v
Dense(29)  --> 899 params
    |
    v
Output (29 features)
```

- **Total parametros entrenables:** 3,594
- **Funcion de activacion:** tanh
- **Funcion de perdida:** MSE (Error Cuadratico Medio)
- **Metrica auxiliar:** MAE (Error Absoluto Medio)

### Estrategia de Deteccion

1. Entrenar el autoencoder unicamente con transacciones legitimas
2. Calcular el error de reconstruccion para cada transaccion
3. Definir umbral = media + 4 * desviacion estandar del error
4. Transacciones con error superior al umbral se clasifican como fraude

---

## Dataset

| Caracteristica | Valor |
|----------------|-------|
| Fuente | [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| Transacciones totales | 284,807 |
| Caracteristicas | 31 (Time, V1-V28, Amount, Class) |
| Transacciones legitimas | 284,315 (99.83%) |
| Transacciones fraudulentas | 492 (0.17%) |

El dataset incluye caracteristicas V1-V28 anonimizadas mediante PCA por razones de confidencialidad.

---

## Resultados

| Metrica | Valor |
|---------|-------|
| Umbral de decision | 0.1362 (media + 4*std) |
| **Recall** | **0.368 (36.8%)** |
| Perdida final (MSE) | ~0.0072 |
| Epochs entrenados | 20 |

**Nota:** El bajo recall indica que el modelo detecta aproximadamente el 37% de los fraudes. Este es un comportamiento esperado en modelos de autoencoder no supervisados, donde se prioriza la especificidad sobre la sensibilidad. El umbral puede ajustarse para mejorar la cobertura.

---

## Estructura del Proyecto

```
Creditcard_fraud_prediction/
|
|-- Creditcard_redneuronal_2025_12_10_rev00.ipynb   # Notebook principal
|-- README.md                                        # Este archivo
|-- requirements.txt                                 # Dependencias (opcional)
```

---

## Como Ejecutar

### Requisitos

- Python 3.8+
- TensorFlow 2.x
- Keras
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn

### Pasos

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/xfalconix/Creditcard_fraud_prediction.git
   cd Creditcard_fraud_prediction
   ```

2. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Obtener el dataset:**
   - Descargar desde [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
   - Colocar el archivo `creditcard.csv` en la ruta esperada o montar Google Drive

4. **Ejecutar en Google Colab (recomendado):**
   - Abrir el notebook en Google Colab
   - Ejecutar las celdas secuencialmente

5. **Ejecutar localmente:**
   ```bash
   jupyter notebook Creditcard_redneuronal_2025_12_10_rev00.ipynb
   ```

### Ejecucion en Google Colab

El notebook incluye un boton para abrirlo directamente en Google Colab. Solo necesitas tener el dataset accesible (Google Drive o carga directa).

---

## Secciones del Notebook

1. **Imports y configuracion** - Librerias necesarias
2. **Carga de datos** - Lectura del CSV
3. **EDA (Exploratory Data Analysis)** - Analisis exploratorio, visualizacion de la distribucion de clases
4. **Preprocesamiento** - Escalado MinMax de las caracteristicas
5. **Construccion del Autoencoder** - Definicion de capas y compilacion
6. **Entrenamiento** - 20 epochs con datos de transacciones legitimas
7. **Evaluacion** - Calculo del umbral y metricas de deteccion

---

## Limitaciones y Mejoras Futuras

- **Umbral fijo:** Considerar optimizacion del umbral mediante curvas ROC
- **Clases desbalanceadas:** Explorar tecnicas como SMOTE o class weighting
- **Red mas profunda:** Experimentar con mas capas o regularizacion
- **Comparacion:** Contrastar con modelos supervisados (XGBoost, Random Forest)

---

## Licencia

MIT License - Ver archivo LICENSE para mas detalles.

---

*Proyecto desarrollado como parte del Master en Big Data, Data Engineering and AI - ESESA - Malaga, Espana - 2025-2026*
