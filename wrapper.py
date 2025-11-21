# wrapper.py

import pandas as pd
from joblib import load

# 1. Diccionarios de mapeo texto -> numérico según documentación UCI

map_age = {
    "18-24": -0.95197,
    "25-34": -0.07854,
    "35-44":  0.49788,
    "45-54":  1.09449,
    "55-64":  1.82213,
    "65+":    2.59171
}

# En la descripción: 0.48246 = Female, -0.48246 = Male
map_gender = {
    "Mujer":   0.48246,
    "Hombre": -0.48246
}

map_education = {
    "Dejó la escuela antes de los 16":          -2.43591,
    "Dejó la escuela a los 16":                 -1.73790,
    "Dejó la escuela a los 17":                 -1.43719,
    "Dejó la escuela a los 18":                 -1.22751,
    "Algo de universidad sin título":           -0.61113,
    "Certificado profesional / diploma":        -0.05921,
    "Grado universitario":                      0.45468,
    "Máster":                                   1.16365,
    "Doctorado":                                1.98437,
}

map_country = {
    "Australia":           -0.09765,
    "Canadá":               0.24923,
    "Nueva Zelanda":       -0.46841,
    "Otros":               -0.28519,
    "República de Irlanda": 0.21128,
    "Reino Unido":          0.96082,
    "Estados Unidos":      -0.57009,
}

map_ethnicity = {
    "Asiático":              -0.50212,
    "Negro":                 -1.10702,
    "Mixto negro/asiático":   1.90725,
    "Mixto blanco/asiático":  0.12600,
    "Mixto blanco/negro":    -0.22166,
    "Otro":                   0.11440,
    "Blanco":                -0.31685,
}


def map_personality_slider(v: int) -> float:
    """
    Convierte un slider 1-10 en un valor aproximado continuo.
    Aquí hacemos un mapeo lineal al rango [-2, 2].
    Si quieres puedes ajustarlo más tarde usando la distribución real.
    """
    return -2 + (v - 1) * (4 / 9)


class CannabisModelWrapper:
    def __init__(self, model, scaler, feature_names, feature_means):
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names
        self.feature_means = feature_means  # dict: {col: media}

    @classmethod
    def load_from_file(cls, path: str) -> "CannabisModelWrapper":
        artefacto = load(path)
        return cls(
            model=artefacto["model"],
            scaler=artefacto["scaler"],
            feature_names=artefacto["features"],
            feature_means=artefacto["feature_means"],
        )

    def build_features(
        self,
        age_text: str,
        gender_text: str,
        education_text: str,
        country_text: str,
        ethnicity_text: str,
        nscore_level: int,
        escore_level: int,
        oscore_level: int,
        ascore_level: int,
        cscore_level: int,
        impuls_level: int,
        sensation_level: int,
    ) -> pd.DataFrame:
        """
        Recibe inputs 'humanos' y devuelve un DataFrame de 1 fila
        con las features en el orden correcto.
        """

        # Partimos de las medias de cada columna (incluye id, etc.)
        vals = self.feature_means.copy()

        # Sobrescribimos las columnas que el usuario controla
        # Usamos TUS nombres reales: 'age', 'gender', 'education', 'country', 'ethnicity',
        # 'nscore', 'escore', 'oscore', 'ascore', 'cscore', 'impuslive', 'ss'.

        vals["age"]        = map_age[age_text]
        vals["gender"]     = map_gender[gender_text]
        vals["education"]  = map_education[education_text]
        vals["country"]    = map_country[country_text]
        vals["ethnicity"]  = map_ethnicity[ethnicity_text]

        vals["nscore"]     = map_personality_slider(nscore_level)
        vals["escore"]     = map_personality_slider(escore_level)
        vals["oscore"]     = map_personality_slider(oscore_level)
        vals["ascore"]     = map_personality_slider(ascore_level)
        vals["cscore"]     = map_personality_slider(cscore_level)
        vals["impuslive"]  = map_personality_slider(impuls_level)
        vals["ss"]         = map_personality_slider(sensation_level)

        # Construimos el dataframe en el orden esperado por el modelo
        row = [vals[col] for col in self.feature_names]
        X_input = pd.DataFrame([row], columns=self.feature_names)
        return X_input

    def predict(self, **kwargs):
        """
        kwargs: los mismos argumentos que build_features.
        Devuelve (label, prob_consumidor).
        """
        X_input = self.build_features(**kwargs)
        X_scaled = self.scaler.transform(X_input)
        proba = self.model.predict_proba(X_scaled)[0, 1]
        label = int(self.model.predict(X_scaled)[0])
        return label, proba
