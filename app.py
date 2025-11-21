# app.py

import streamlit as st
from wrapper import CannabisModelWrapper

st.title("Predicción de consumo de cannabis")

@st.cache_resource
def load_model():
    return CannabisModelWrapper.load_from_file("modelo_cannabis.pkl")

wrapper = load_model()

st.subheader("Perfil de la persona")

age_text = st.selectbox(
    "Grupo de edad",
    ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
)

gender_text = st.selectbox(
    "Género",
    ["Hombre", "Mujer"]
)

education_text = st.selectbox(
    "Nivel educativo",
    [
        "Dejó la escuela antes de los 16",
        "Dejó la escuela a los 16",
        "Dejó la escuela a los 17",
        "Dejó la escuela a los 18",
        "Algo de universidad sin título",
        "Certificado profesional / diploma",
        "Grado universitario",
        "Máster",
        "Doctorado",
    ]
)

country_text = st.selectbox(
    "País de residencia",
    [
        "Australia",
        "Canadá",
        "Nueva Zelanda",
        "Otros",
        "República de Irlanda",
        "Reino Unido",
        "Estados Unidos",
    ]
)

ethnicity_text = st.selectbox(
    "Etnia",
    [
        "Asiático",
        "Negro",
        "Mixto negro/asiático",
        "Mixto blanco/asiático",
        "Mixto blanco/negro",
        "Otro",
        "Blanco",
    ]
)

st.markdown("### Rasgos de personalidad (1 = bajo, 10 = alto)")

nscore_level = st.slider("Neuroticismo", 1, 10, 5)
escore_level = st.slider("Extraversión", 1, 10, 5)
oscore_level = st.slider("Apertura a la experiencia", 1, 10, 5)
ascore_level = st.slider("Amabilidad", 1, 10, 5)
cscore_level = st.slider("Responsabilidad / escrupulosidad", 1, 10, 5)
impuls_level = st.slider("Impulsividad", 1, 10, 5)
sensation_level = st.slider("Búsqueda de sensaciones", 1, 10, 5)

if st.button("Predecir consumo de cannabis"):
    label, proba = wrapper.predict(
        age_text=age_text,
        gender_text=gender_text,
        education_text=education_text,
        country_text=country_text,
        ethnicity_text=ethnicity_text,
        nscore_level=nscore_level,
        escore_level=escore_level,
        oscore_level=oscore_level,
        ascore_level=ascore_level,
        cscore_level=cscore_level,
        impuls_level=impuls_level,
        sensation_level=sensation_level,
    )

    st.write(f"Probabilidad estimada de ser consumidor de cannabis: {proba:.2%}")
    if label == 1:
        st.warning("El modelo clasifica este perfil como CONSUMIDOR.")
    else:
        st.success("El modelo clasifica este perfil como NO consumidor.")
