Predicción de Consumo de Cannabis – Proyecto de Machine Learning

Este proyecto analiza un dataset real del repositorio UCI para predecir si una persona tiene mayor probabilidad de consumir cannabis.
La predicción se basa en variables demográficas y rasgos de personalidad, utilizando distintos modelos de Machine Learning y una aplicación interactiva en Streamlit.

1. Objetivos del proyecto

Analizar la relación entre perfil personal y consumo de cannabis.

Convertir el dataset original en un problema de clasificación binaria.

Probar y comparar varios modelos de Machine Learning.

Optimizar los modelos con ajuste de hiperparámetros.

Crear una aplicación interactiva que permita generar predicciones a partir de un perfil introducido por el usuario.

2. Dataset

Fuente: UCI Machine Learning Repository – Drug Consumption (Quantified)
Tamaño: ~1.900 participantes

Variables utilizadas

Datos demográficos

Edad

Género

Nivel educativo

País

Etnia

Rasgos de personalidad

Neuroticismo (N)

Extraversión (E)

Apertura mental (O)

Amabilidad (A)

Responsabilidad / Escrupulosidad (C)

Impulsividad (Imp)

Búsqueda de sensaciones (SS)

Variable objetivo

Transformación de la variable original de consumo de cannabis a binaria:

0 = No consumidor (CL0, CL1, CL2)

1 = Consumidor (CL3, CL4, CL5, CL6)

3. Preparación de datos

Eliminación de duplicados

Comprobación de tipos y consistencia

Transformación del target a formato binario

Separación en variables predictoras (X) y variable objetivo (y)

Normalización para modelos que lo requieren (como KNN)

4. Modelos evaluados

KNN

Regresión logística

Árbol de decisión

Bagging

Random Forest

AdaBoost

Gradient Boosting

Métricas utilizadas

Accuracy

F1-score

Ambas métricas se usaron para garantizar un equilibrio entre predicción de consumidores y no consumidores.

5. Ajuste de hiperparámetros

Los dos modelos con mejor rendimiento inicial fueron:

Random Forest

Gradient Boosting

A ambos se les aplicó una búsqueda exhaustiva de hiperparámetros probando un total de 63 combinaciones.

Ejemplos de parámetros probados:

Random Forest

n_estimators: 100, 200, 300

max_depth: 3, 5, 7, None

min_samples_split: 2, 5, 10

Gradient Boosting

n_estimators: 50, 100, 150

learning_rate: 0.01, 0.05, 0.1

max_depth: 2, 3, 4

Mejora obtenida tras la optimización:

Random Forest: +0.024 en accuracy

Gradient Boosting: +0.019 en accuracy

Ligera mejora adicional en F1-score

El mejor modelo global fue el Gradient Boosting optimizado.

6. Aplicación interactiva en Streamlit

El proyecto incluye una aplicación que permite:

Introducir un perfil demográfico

Ajustar rasgos de personalidad mediante barras deslizantes

Generar una predicción en tiempo real

Mostrar probabilidad estimada y clasificación final

Ejecutar la app:
streamlit run app.py

Ejecutar la presentación:
streamlit run presentation.py

7. Estructura del proyecto
proyecto-cannabis-ml/
│
├── app.py                 # Aplicación interactiva en Streamlit
├── presentation.py        # Presentación del proyecto en modo diapositivas
├── wrapper.py             # Funciones auxiliares (opcional)
├── README.md
└── requirements.txt       # Dependencias del proyecto

8. Instalación

Instalar dependencias:

pip install -r requirements.txt


Dependencias principales:

streamlit

pandas

scikit-learn

altair

ucimlrepo

9. Conclusiones

Los datos demográficos y de personalidad permiten estimar con cierta fiabilidad el consumo de cannabis.

Gradient Boosting optimizado fue el modelo más preciso y equilibrado.

La definición del target y el uso adecuado de las features influyeron más en el rendimiento que la complejidad del modelo.

10. Futuras mejoras

Analizar grados de consumo en lugar de clasificación binaria.

Añadir interpretabilidad mediante SHAP.

Incorporar modelos adicionales para comparación.

Analizar consumo conjunto de múltiples sustancias.

Autor

Elías Chafih Meddah
Data & Business Intelligence Analyst
LinkedIn: (añadir enlace)
GitHub: (añadir enlace)