# Red Neuronal Autoencoder para Deteccion de Fraude en Tarjetas de Credito

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=flat-square&logo=kaggle&logoColor=white)
![MIT](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

## Descripcion

Este proyecto implementa un **Autoencoder** construido desde cero con **NumPy puro** — sin frameworks de deep learning — para la deteccion de transacciones fraudulentas en tarjetas de credito.

El modelo aprende a reconstruir patrones de transacciones legitimas; aquellas con alto error de reconstruccion se clasifican como anomalias potenciales (fraudes).

**Autor:** Carlos Falconi — Malaga, Espana  
**Fecha:** Diciembre 2025  
**Contexto:** Master en Big Data, Data Engineering & AI — ESESA, Malaga 2025-2026

---

## Enfoque Tecnico

### Arquitectura del Autoencoder (implementacion NumPy pura)

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
- **Optimizacion:** Gradiente descendente con backpropagation manual
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

Las caracteristicas V1-V28 estan anonimizadas mediante PCA por razones de confidencialidad.

---

## Resultados

| Metrica | Valor |
|---------|-------|
| Umbral de decision | 0.1362 (media + 4*std) |
| **Recall** | **36.8%** |
| Perdida final (MSE) | ~0.0072 |
| Epochs entrenados | 20 |

El bajo recall es un comportamiento esperado en autoencoders no supervisados: al entrenarse solo con transacciones legitimas, optimizan la reconstruccion del patron normal, y las anomalias elevan naturalmente el error. El umbral es conservativo para minimizar falsos positivos. Para mejorar la cobertura bastaria bajar el umbral.

---

## Estructura del Proyecto

```
fraud-detection-api/
|
|-- Creditcard_redneuronal_2025_12_10_rev00.ipynb   # Notebook principal
|-- README.md                                        # Este archivo
```

---

## Como Ejecutar

### Requisitos

- Python 3.8+
- NumPy
- Pandas
- Matplotlib, Seaborn
- scikit-learn (para MinMaxScaler y metricas)

### Pasos

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/xfalconix/fraud-detection-api.git
   cd fraud-detection-api
   ```

2. **Instalar dependencias:**
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn jupyter
   ```

3. **Obtener el dataset:**
   - Descargar desde [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
   - Colocar el archivo `creditcard.csv` en el directorio del notebook

4. **Ejecutar en Google Colab (recomendado):**
   - Abrir el notebook en GitHub y pulsar el boton **Open in Colab**
   - Montar Google Drive o subir el CSV directamente

5. **Ejecutar localmente:**
   ```bash
   jupyter notebook Creditcard_redneuronal_2025_12_10_rev00.ipynb
   ```

---

## Secciones del Notebook

| Seccion | Contenido |
|---------|-----------|
| Imports y configuracion | NumPy, Pandas, Matplotlib, Seaborn, sklearn |
| Carga de datos | Lectura del CSV de Kaggle |
| EDA | Distribucion de clases, visualizacion de V1-V28 |
| Preprocesamiento | MinMaxScaler sobre las 29 features |
| Construccion del Autoencoder | Capas densas con inicializacion Xavier, tanh |
| Entrenamiento | 20 epochs con backpropagation manual sobre solo transacciones legitimas |
| Evaluacion | Calculo de umbral, error de reconstruccion, recall |

---

## Limitaciones y Mejoras Futuras

- **Umbral fijo:** Optimizar el umbral con curvas ROC o precision-recall
- **Clases desbalanceadas:** Experimentar con SMOTE, undersampling o class weighting
- **Arquitectura mas profunda:** Probar mas capas o regularizacion L2
- **Comparacion supervisada:** Contrastar con XGBoost o Random Forest supervisados
- **Despliegue:** Exportar el modelo y servir como API con FastAPI

---

## Por que NumPy puro?

Implementar el autoencoder desde cero (sin TensorFlow/Keras/PyTorch) es un ejercicio pedagogico fundamental para entender:
- Como funciona el forward pass y backpropagation a nivel matematico
- Como se propagan gradientes a traves de capas densas
- La diferencia entre usar un framework y entender lo que hace por debajo

Esto demuestra dominio profundo de los fundamentos de deep learning — algo que los recrutadores valoran especialmente en perfiles junior-to-mid en ML.

---

*Proyecto desarrollado como parte del Master en Big Data, Data Engineering & AI — ESESA, Malaga 2025-2026*
