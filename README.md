<div align="center">

# Predicción de Consumo de Cannabis  
### Proyecto de Machine Learning con datos del UCI Machine Learning Repository

---

</div>

## Índice

1. [Descripción general](#descripción-general)  
2. [Dataset](#dataset)  
3. [Preparación de datos](#preparación-de-datos)  
4. [Modelos y métricas](#modelos-y-métricas)  
5. [Ajuste de hiperparámetros](#ajuste-de-hiperparámetros)  
6. [Resultados principales](#resultados-principales)  
7. [Aplicación en Streamlit](#aplicación-en-streamlit)  
8. [Instalación](#instalación)  
9. [Estructura del proyecto](#estructura-del-proyecto)  
10. [Futuras mejoras](#futuras-mejoras)  
11. [Licencia](#licencia)  

---

## Descripción general

Este proyecto desarrolla un modelo de **clasificación binaria** para estimar si una persona tiene mayor probabilidad de consumir cannabis a partir de:

- Variables demográficas  
- Rasgos de personalidad cuantificados  

El trabajo incluye:

- Análisis y preparación del dataset  
- Selección de variables de entrada (features)  
- Comparación de varios algoritmos de Machine Learning  
- Ajuste de hiperparámetros de los mejores modelos  
- Desarrollo de una **aplicación interactiva** con Streamlit  
- Una **presentación tipo diapositivas** también en Streamlit

---

## Dataset

**Fuente**: UCI Machine Learning Repository  
**Nombre original**: *Drug Consumption (Quantified)*  
**Número aproximado de registros**: 1.900 personas

### Variables utilizadas

**Demografía**

- Edad  
- Género  
- Nivel educativo  
- País  
- Etnia  

**Rasgos de personalidad (escala numérica)**

- Neuroticismo (N)  
- Extraversión (E)  
- Apertura mental (O)  
- Amabilidad (A)  
- Responsabilidad / Escrupulosidad (C)  
- Impulsividad (Imp)  
- Búsqueda de sensaciones (SS)  

**Variable objetivo (target)**

Se transforma el consumo original de cannabis en una variable binaria:

- `0` → No consumidor (consumo nulo o muy esporádico)  
- `1` → Consumidor (consumo más frecuente)

---

## Preparación de datos

Principales pasos realizados:

1. **Limpieza básica**
   - Eliminación de posibles duplicados  
   - Revisión de tipos de datos y valores ausentes

2. **Construcción de la variable objetivo**
   - Conversión del nivel de consumo de cannabis en una variable binaria (`cannabis_binary`)

3. **Selección de variables**
   - Uso únicamente de:
     - Variables demográficas  
     - Rasgos de personalidad  
   - Exclusión explícita del consumo de otras sustancias como predictor (el objetivo es basarse en el perfil, no en consumos previos)

4. **Normalización**
   - Aplicada en los modelos sensibles a la escala (por ejemplo, KNN), para hacer comparables las magnitudes de las variables

---

## Modelos y métricas

Se entrenaron y compararon diferentes modelos de clasificación:

| Modelo                    | Tipo general                           |
|---------------------------|----------------------------------------|
| KNN                       | Vecinos más cercanos                   |
| Regresión logística      | Modelo lineal probabilístico           |
| Árbol de decisión         | Árbol de clasificación                 |
| Bagging                   | Conjunto de árboles (bootstrap)       |
| Random Forest            | Bosque aleatorio de árboles            |
| AdaBoost                 | Ensamblado aditivo secuencial          |
| Gradient Boosting        | Ensamblado aditivo con gradiente       |

**Métricas utilizadas:**

- **Accuracy**: porcentaje de aciertos totales  
- **F1-score**: equilibrio entre precisión y exhaustividad, útil para asegurar un buen comportamiento tanto en consumidores como en no consumidores

---

## Ajuste de hiperparámetros

Tras la comparación inicial, se seleccionaron dos modelos para afinarlos:

- **Random Forest**  
- **Gradient Boosting**

Se evaluaron distintas combinaciones de hiperparámetros mediante búsqueda sistemática.

### Hiperparámetros explorados

**Random Forest**

- `n_estimators`: 100, 200, 300  
- `max_depth`: 3, 5, 7, None  
- `min_samples_split`: 2, 5, 10  

Combinaciones evaluadas:  
`3 (n_estimators) × 4 (max_depth) × 3 (min_samples_split) = 36`

**Gradient Boosting**

- `n_estimators`: 50, 100, 150  
- `learning_rate`: 0.01, 0.05, 0.1  
- `max_depth`: 2, 3, 4  

Combinaciones evaluadas:  
`3 × 3 × 3 = 27`

**Total configuraciones probadas**:  
`36 + 27 = 63`

### Efecto del ajuste

- Mejora en **accuracy** de aproximadamente:
  - Random Forest: +0,024  
  - Gradient Boosting: +0,019  
- Ligera mejora adicional en **F1-score**, indicando un mejor equilibrio entre ambas clases

El modelo final con mejor rendimiento global fue **Gradient Boosting ajustado**.

---

## Resultados principales

- Los mejores modelos alcanzan una **accuracy superior al 83 %**.  
- El **F1-score** indica que el modelo mantiene un buen equilibrio entre predicción de consumidores y no consumidores.  
- Se observa que la combinación de variables de personalidad y datos demográficos aporta valor para la predicción.  
- La preparación del objetivo y la correcta selección de variables tienen un impacto significativo en el rendimiento, incluso más que la complejidad del modelo.

---

## Aplicación en Streamlit

El repositorio incluye:

1. Una **aplicación interactiva** (`app.py`) donde el usuario puede:
   - Introducir un perfil demográfico  
   - Ajustar los rasgos de personalidad mediante controles deslizantes  
   - Obtener una predicción y una probabilidad asociada

2. Una **presentación interactiva** (`presentation.py`) que explica el proyecto en formato de diapositivas, también en Streamlit.

### Ejecutar la aplicación

```bash
streamlit run app.py
