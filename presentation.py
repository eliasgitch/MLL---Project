import streamlit as st
import pandas as pd
import altair as alt
from ucimlrepo import fetch_ucirepo

st.set_page_config(
    page_title="Predicción de consumo de cannabis",
    layout="wide"
)

# ==== ESTILO GLOBAL (CSS) ====
st.markdown(
    """
    <style>

    /* Forzar que el título sea realmente gigante */
   .big-title, .big-title * {
      font-size: 80px !important;
      font-weight: 900 !important;
      line-height: 1.05 !important;
      text-align: center !important;
    }

    .subtitle {
        font-size: 26px;            /* Un poco más grande también */
        color: #444;
        margin-bottom: 1.8rem;
        text-align: center;
    }

    /* === QUITAR FONDOS OSCUROS === */
    .hero-box {
        padding: 1.8rem 2.2rem;
        border-radius: 1.2rem;
        background: #ffffff;
        color: #222;
        border: 1px solid #e0e0e0;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 20px rgba(0,0,0,0.08);
    }

    .section-box {
        padding: 1.2rem 1.5rem;
        border-radius: 0.8rem;
        background-color: #fafafa;
        border: 1px solid #e5e5e5;
        margin-bottom: 1rem;
    }

    /* Badges */
    .badge {
        display: inline-block;
        padding: 0.3rem 0.9rem;
        border-radius: 999px;
        background-color: #000;
        color: white;
        font-size: 0.85rem;
        margin-right: 0.3rem;
        margin-bottom: 0.3rem;
    }

    /* Forzar fondo general blanco */
    body, .stApp {
        background-color: white !important;
        color: black !important;
    }

    h1, h2, h3, h4, h5, h6, p, div {
        color: black !important;
    }

    /* ======== SIDEBAR CLARO ======== */
    [data-testid="stSidebar"] {
        background-color: #f4f4f4 !important;    /* Fondo claro */
        color: #000 !important;                  /* Texto negro */
        border-right: 1px solid #ddd !important;
    }

    /* Texto dentro del sidebar */
    [data-testid="stSidebar"] * {
        color: #000 !important;
    }

    /* Radio buttons dentro del sidebar */
    [data-testid="stSidebar"] .stRadio > div > label {
        color: #000 !important;
    }

    /* Títulos del sidebar */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #000 !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==== CARGA DEL DATASET PARA EDA ====
@st.cache_data
def load_dataset():
    drug = fetch_ucirepo(id=373)
    df = drug.data.original.copy()
    # crear target binario igual que en el notebook
    df["cannabis_binary"] = df["cannabis"].apply(
        lambda x: 0 if x in ["CL0", "CL1", "CL2"] else 1
    )
    return df

df = load_dataset()

# --------- SIDEBAR: ÍNDICE DE DIAPOSITIVAS ---------
slides = [
    "1. Título",
    "2. Descripción general + EDA",
    "3. Datos, preparación y selección",
    "4. Modelos y métricas",
    "5. Hiperparámetros",
    "6. Conclusiones",
    "7. Aplicación real",
    "8. Demo interactiva",
    "10. Futuro",
]

st.sidebar.title("Índice")
current_slide = st.sidebar.radio("Ir a sección:", slides)


# --------- DIAPOSITIVAS ---------

def slide_1():
    st.markdown(
        '<p class="big-title">Predicción de consumo de cannabis</p>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p class="subtitle">Proyecto de Machine Learning</p>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="hero-box">
        <strong>¿En qué consiste este proyecto?</strong><br><br>
        Hemos utilizado un conjunto de datos de casi 2.000 personas con información
        básica sobre su perfil:
        <ul>
            <li>Edad, género, nivel de estudios, país y origen étnico.</li>
            <li>Rasgos de personalidad (por ejemplo, si alguien es más impulsivo o más responsable).</li>
        </ul>
        Con esa información entrenamos un modelo de machine learning que intenta responder:
        <br><br>
        <em>'Dado un perfil, ¿es más probable que la persona consuma o no cannabis?'</em>
        </div>
        """,
        unsafe_allow_html=True
    )


def slide_2():
    st.header("Descripción general del proyecto + EDA rápida")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class="section-box">
            <strong>De dónde salen los datos</strong>
            <ul>
              <li>Dataset público: <em>Drug Consumption (Quantified)</em> del repositorio UCI.</li>
              <li>Cada fila es una persona que respondió a un cuestionario.</li>
              <li>Para este proyecto nos centramos en el consumo de cannabis.</li>
            </ul>
            <strong>Qué queremos predecir</strong>
            <ul>
              <li>Convertimos el consumo en una pregunta de sí/no:
                <ul>
                  <li>0 = no consumidor (o consumo muy raro).</li>
                  <li>1 = consumidor (consumo más frecuente).</li>
                </ul>
              </li>
              <li>Usamos el perfil psicométrico y demográfico como entrada del modelo.</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class="section-box">
            <strong>Foto rápida del conjunto de datos</strong>
            <ul>
              <li>Aproximadamente 1.900 participantes.</li>
              <li>Género bastante equilibrado.</li>
              <li>Principalmente gente entre 18 y 44 años.</li>
              <li>Los datos vienen ya 'limpios' y en formato numérico.</li>
              <li>Las clases (consumidor / no consumidor) están razonablemente equilibradas.</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Distribución del target")
        target_counts = df["cannabis_binary"].value_counts().sort_index()
        target_df = target_counts.rename_axis("Clase").reset_index(name="Recuentos")
        chart_target = (
            alt.Chart(target_df)
            .mark_bar()
            .encode(
                x=alt.X("Clase:N", axis=alt.Axis(title="0 = No consumidor, 1 = Consumidor")),
                y=alt.Y("Recuentos:Q"),
                tooltip=["Clase", "Recuentos"]
            )
            .properties(height=300)
        )
        st.altair_chart(chart_target, use_container_width=True)

    # Segunda columna vacía (por si quieres añadir algo luego)
    with col4:
        st.write("")


def slide_3_4():
    st.header("Datos, preparación y selección de variables")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class="section-box">
            <strong>Datos utilizados en el proyecto</strong>
            <ul>
              <li>Dataset público del repositorio UCI con alrededor de 1.900 personas.</li>
              <li>Incluye información demográfica y 7 rasgos de personalidad NEO PI-R (90s).</li>
              <li>Los datos vienen totalmente numéricos, sin necesidad de recodificación compleja.</li>
              <li>Eliminamos duplicados y comprobamos que no faltaran datos relevantes.</li>
            </ul>

            <strong>Definición del objetivo</strong>
            <ul>
              <li>Transformamos la variable original de consumo de cannabis en un valor binario:</li>
              <ul>
                <li>0 → No consumidor (o consumo muy esporádico)</li>
                <li>1 → Consumidor (consumo más frecuente)</li>
              </ul>
              <li>Esta decisión define claramente qué queremos que el modelo aprenda.</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class="section-box">
            <strong>Variables seleccionadas como entrada del modelo</strong>
            <ul>
              <li><strong>Datos demográficos:</strong>
                <ul>
                  <li>Edad</li>
                  <li>Género</li>
                  <li>Nivel educativo</li>
                  <li>País</li>
                  <li>Origen étnico</li>
                </ul>
              </li>

              <li><strong>Rasgos de personalidad:</strong>
                <ul>
                  <li>Neuroticismo</li>
                  <li>Extraversión</li>
                  <li>Apertura mental</li>
                  <li>Amabilidad</li>
                  <li>Responsabilidad / Escrupulosidad</li>
                  <li>Impulsividad</li>
                  <li>Búsqueda de sensaciones</li>
                </ul>
              </li>

              <li><strong>Decisiones importantes:</strong>
                <ul>
                  <li>No usamos el consumo de otras drogas como predictor,
                      para basarnos únicamente en el perfil de la persona.</li>
                  <li>Aplicamos normalización solo en los modelos que lo necesitaban
                      (por ejemplo, KNN), para que todas las variables tengan escalas comparables.</li>
                </ul>
              </li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True
        )


def slide_5():
    st.header("Modelos y métricas")

    st.markdown(
        """
        <span class="badge">KNN</span>
        <span class="badge">Regresión logística</span>
        <span class="badge">Árbol de decisión</span>
        <span class="badge">Bagging</span>
        <span class="badge">Random Forest</span>
        <span class="badge">AdaBoost</span>
        <span class="badge">Gradient Boosting</span>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="section-box">
        <strong>Cómo evaluamos los modelos</strong>
        <ul>
          <li>Dividimos los datos en dos grupos:
            <ul>
              <li><strong>Entrenamiento</strong> (80 %): para que el modelo "aprenda".</li>
              <li><strong>Prueba</strong> (20 %): para comprobar cómo se comporta con personas que no ha visto.</li>
            </ul>
          </li>
          <li>Métrica principal: <strong>Accuracy</strong>, es decir,
              el porcentaje de personas que el modelo clasifica bien.</li>
          <li>También miramos el <strong>F1-score</strong>, que resume cómo de bien
              acierta tanto con los consumidores como con los no consumidores al mismo tiempo.</li>
          <li>Combinamos <em>accuracy</em> y <em>F1-score</em> para decidir qué modelo es más equilibrado.</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

    # accuracies reales
    acc_data = {
        "Modelo": [
            "KNN",
            "Regresión logística",
            "Árbol de decisión",
            "Bagging",
            "Random Forest (tuned)",
            "AdaBoost",
            "Gradient Boosting (tuned)",
        ],
        "Accuracy": [
            0.8011,
            0.8302,
            0.7560,
            0.8090,
            0.8329,
            0.8170,
            0.8355,
        ],
    }
    df_acc = pd.DataFrame(acc_data)

    col1, col2 = st.columns([1, 1.3])

    with col1:
        st.markdown("**Porcentaje de aciertos en el conjunto de prueba**")
        st.dataframe(df_acc.style.format({"Accuracy": "{:.4f}"}), use_container_width=True)
        st.markdown(
            """
            En general, los modelos que salen mejor en <strong>accuracy</strong> también
            mantienen un <strong>F1-score alto</strong>, es decir, equilibran bien los aciertos
            en ambos grupos (consumidores y no consumidores).
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown("**Comparativa visual de modelos (accuracy)**")
        chart = (
            alt.Chart(df_acc)
            .mark_bar()
            .encode(
                x=alt.X("Modelo:N", sort=None, axis=alt.Axis(labelAngle=-45)),
                y=alt.Y("Accuracy:Q"),
                tooltip=["Modelo", "Accuracy"]
            )
            .properties(
                width=600,
                height=400
            )
        )
        st.altair_chart(chart, use_container_width=True)


def slide_6():
    st.header("Ajuste de hiperparámetros")

    st.markdown(
        """
        <div class="section-box">
        <strong>Qué hicimos exactamente en esta fase</strong>
        <ul>
          <li>Tras comparar todos los modelos, centramos la optimización en los dos mejores:
            <ul>
              <li><strong>Random Forest</strong></li>
              <li><strong>Gradient Boosting</strong></li>
            </ul>
          </li>
          <li>Definimos una búsqueda sistemática de hiperparámetros usando combinaciones predefinidas.</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="section-box">
        <strong>Hiperparámetros probados</strong>

        <p><strong>Random Forest</strong></p>
        <ul>
          <li><strong>n_estimators</strong> (número de árboles): 100, 200, 300</li>
          <li><strong>max_depth</strong>: 3, 5, 7, None</li>
          <li><strong>min_samples_split</strong>: 2, 5, 10</li>
          <li><strong>Total combinaciones evaluadas:</strong> 3 × 4 × 3 = <strong>36</strong></li>
        </ul>

        <p><strong>Gradient Boosting</strong></p>
        <ul>
          <li><strong>n_estimators</strong>: 50, 100, 150</li>
          <li><strong>learning_rate</strong>: 0.01, 0.05, 0.1</li>
          <li><strong>max_depth</strong>: 2, 3, 4</li>
          <li><strong>Total combinaciones evaluadas:</strong> 3 × 3 × 3 = <strong>27</strong></li>
        </ul>

        <p><strong>Total combinaciones exploradas:</strong> 36 + 27 = <strong>63 configuraciones distintas</strong></p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="section-box">
        <strong>Resultados numéricos de la optimización</strong>
        <ul>
          <li>Antes del ajuste:
            <ul>
              <li><strong>Random Forest</strong>: Accuracy ≈ 0.809</li>
              <li><strong>Gradient Boosting</strong>: Accuracy ≈ 0.817</li>
            </ul>
          </li>
          <li>Después del ajuste:
            <ul>
              <li><strong>Random Forest (tuned)</strong>: Accuracy ≈ 0.833</li>
              <li><strong>Gradient Boosting (tuned)</strong>: Accuracy ≈ 0.836</li>
            </ul>
          </li>
          <li><strong>Mejora lograda:</strong>
            <ul>
              <li>Random Forest: +0.024 puntos de accuracy</li>
              <li>Gradient Boosting: +0.019 puntos de accuracy</li>
            </ul>
          </li>
          <li>Además, ambos modelos aumentaron ligeramente su <strong>F1-score</strong>, indicando mejores predicciones equilibradas.</li>
        </ul>

        <p>En resumen: exploramos 63 combinaciones y nos quedamos con las que proporcionaron el mejor equilibrio entre accuracy y F1.</p>
        </div>
        """,
        unsafe_allow_html=True
    )



def slide_7():
    st.header("Conclusiones principales")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class="section-box">
            <strong>Qué modelos funcionan mejor</strong>
            <ul>
              <li><strong>Gradient Boosting (ajustado)</strong> es el modelo que mejor resultado consigue,
                  con algo más de un 83 % de aciertos.</li>
              <li><strong>Random Forest (ajustado)</strong> y la <strong>regresión logística</strong> quedan muy cerca.</li>
              <li>KNN, Bagging y AdaBoost dan resultados correctos, aunque un peldaño por debajo.</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class="section-box">
            <strong>Qué nos llevamos de todo esto</strong>
            <ul>
              <li>Con un perfil básico de una persona (edad, estudios, personalidad, etc.)
                  se puede estimar con <strong>cierta fiabilidad</strong> si es más probable que consuma cannabis.</li>
              <li>Los modelos basados en <strong>muchos árboles combinados</strong> suelen capturar mejor
                  las relaciones complejas entre variables.</li>
              <li>Un modelo "sencillo" como la <strong>regresión logística</strong>, bien trabajado, sigue siendo muy <strong>competitivo</strong>.</li>
              <li>No basta con elegir un algoritmo potente: la forma de preparar los datos
                  y de <strong>definir</strong> qué es <strong>"consumidor"</strong> marca gran parte del resultado.</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True
        )


def slide_8():
    st.header("Aplicación en el mundo real")

    st.markdown(
        """
        <div class="section-box">
        <strong>Posibles usos</strong>
        <ul>
          <li><strong>Educación y divulgación</strong>:
            <ul>
              <li>Mostrar cómo los datos pueden ayudar a entender patrones de comportamiento.</li>
              <li>Explorar de forma interactiva cómo cambia la probabilidad según el perfil.</li>
            </ul>
          </li>
          <li><strong>Prevención y salud pública</strong>:
            <ul>
              <li>Detectar, a nivel agregado, qué tipos de perfiles podrían requerir más atención preventiva.</li>
            </ul>
          </li>
          <li><strong>Investigación</strong>:
            <ul>
              <li>Estudiar la relación entre personalidad y consumo de sustancias desde un punto de vista estadístico.</li>
            </ul>
          </li>
        </ul>
        <p>
        
        </p>
        </div>
        """,
        unsafe_allow_html=True
    )


def slide_9_demo():
    st.header("Demo interactiva: app de predicción en Streamlit")

    st.markdown(
        """
        <div class='section-box'>
        <strong>¿Qué puedes hacer en la demo?</strong>
        <ul>
          <li>Rellenar un perfil sencillo:
            <ul>
              <li>Grupo de edad, género, nivel educativo, país y etnia.</li>
              <li>Niveles de distintos rasgos de personalidad mediante barras deslizantes.</li>
            </ul>
          </li>
          <li>Al pulsar el botón, la app:
            <ul>
              <li>Transforma tus respuestas en el formato numérico que entiende el modelo.</li>
              <li>Calcula la probabilidad de que ese perfil sea consumidor de cannabis.</li>
              <li>Muestra un mensaje claro: 'consumidor' o 'no consumidor' según el resultado.</li>
            </ul>
          </li>
        </ul>
        <p>
        
        </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class='section-box'>
        <strong>Enlace a la app</strong>
        <p>
            <a href='http://localhost:8503/' target='_blank'>
            Pulsa aquí para simular un perfil
            </a>
       
        </p>
        </div>
        """,
        unsafe_allow_html=True
    )



def slide_10():
    st.header("Futuro y posibles mejoras")

    st.markdown(
        """
        <div class="section-box">
        <strong>Mejoras técnicas sencillas</strong>
        <ul>
          <li>Probar otras formas de medir el <strong>rendimiento</strong>, más allá del porcentaje de aciertos.</li>
          <li>Profundizar en qué variables <strong>influyen más</strong> en la predicción para hacerlo más interpretable.</li>
          <li><strong>Comparar</strong> con algún modelo adicional si se dispone de más tiempo o <strong>más datos</strong>.</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="section-box">
        <strong>Posibles extensiones del proyecto</strong>
        <ul>
          <li>Pasar de una respuesta sí/no a distintos niveles de <strong>frecuencia</strong> de consumo.</li>
          <li>Estudiar el consumo conjunto de <strong>varias</strong> sustancias a la vez.</li>
          <li>Integrar esta demo en un panel más amplio con otros <strong>indicadores de salud</strong> o comportamiento.</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True
    )


# --------- ROUTER DE DIAPOSITIVAS ---------
if current_slide == "1. Título":
    slide_1()
elif current_slide == "2. Descripción general + EDA":
    slide_2()
elif current_slide == "3. Datos, preparación y selección":
    slide_3_4()
elif current_slide == "4. Modelos y métricas":
    slide_5()
elif current_slide == "5. Hiperparámetros":
    slide_6()
elif current_slide == "6. Conclusiones":
    slide_7()
elif current_slide == "7. Aplicación real":
    slide_8()
elif current_slide == "8. Demo interactiva":
    slide_9_demo()
elif current_slide == "10. Futuro":
    slide_10()
