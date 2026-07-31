from __future__ import annotations

import hashlib
import hmac
import os

import requests
import streamlit as st


# URL de la API (Render) o localhost para pruebas.
API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000",
)


st.set_page_config(
    page_title="Sistema de Alerta Temprana",
    page_icon="📘",
    layout="centered",
)


def verificar_acceso() -> None:
    """
    Solicita una contraseña antes de permitir
    el acceso al Sistema de Alerta Temprana.
    """

    password_correcta = os.getenv("APP_PASSWORD")

    if password_correcta is None:
        st.error(
            "No se configuró la contraseña del sistema."
        )
        st.stop()

    if st.session_state.get("autenticado", False):
        return

    st.title("Sistema de Alerta Temprana")
    st.caption(
        "Acceso restringido."
    )

    password = st.text_input(
        "Contraseña",
        type="password",
    )

    if st.button("Ingresar"):

        acceso = hmac.compare_digest(
            hashlib.sha256(
                password.encode("utf-8")
            ).hexdigest(),
            hashlib.sha256(
                password_correcta.encode("utf-8")
            ).hexdigest(),
        )

        if acceso:
            st.session_state["autenticado"] = True
            st.rerun()
        else:
            st.error(
                "Contraseña incorrecta."
            )

    st.stop()


verificar_acceso()


st.title("Sistema de Alerta Temprana")
st.caption(
    "Herramienta de apoyo para identificar posibles "
    "necesidades de acompañamiento educativo."
)


@st.cache_data
def cargar_esquema() -> dict:
    """
    Consulta las variables y categorías permitidas
    directamente desde la API.
    """

    response = requests.get(
        f"{API_URL}/schema",
        timeout=15,
    )
    response.raise_for_status()

    return response.json()

try:
    schema = cargar_esquema()

except requests.RequestException:
    st.error(
        "No fue posible conectarse con la API. "
        "Verifica que FastAPI esté ejecutándose."
    )
    st.stop()


st.success(
    f"Modelo disponible con "
    f"{schema['number_of_features']} variables."
)


payload = {}

with st.form("formulario_prediccion"):

    for feature in schema["feature_order"]:

        # Variables categóricas:
        # se muestran únicamente las categorías aprendidas.
        if feature in schema["categorical_features"]:

            options = schema["categories"].get(
                feature,
                [],
            )

            if not options:
                st.error(
                    f"No se encontraron categorías "
                    f"para {feature}."
                )
                st.stop()

            payload[feature] = st.selectbox(
                label=feature,
                options=options,
                key=feature,
            )

        # Variable numérica discreta del clúster.
        elif feature == "cluster_rendimiento_gmm":

            payload[feature] = st.selectbox(
                label="Grupo de rendimiento",
                options=list(range(7)),
                key=feature,
            )

        else:
            payload[feature] = st.number_input(
                label=feature,
                key=feature,
            )

    submitted = st.form_submit_button(
        "Evaluar estudiante"
    )


if submitted:

    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        result = response.json()
        st.json(result)

    except requests.RequestException as exc:
        st.error(
            "No fue posible generar la predicción."
        )
        st.exception(exc)

    else:
        st.divider()

        st.subheader(
            result["resultado"]
        )

        st.metric(
            "Probabilidad estimada",
            f"{result['probabilidad_estimada']:.1%}",
        )

        factores = result.get(
            "factores",
            [],
        )

        if factores:
            st.markdown(
                "### Aspectos que podrían requerir acompañamiento"
            )

            for factor in factores:
                st.markdown(
                    f"**{factor['dimension']}**  \n"
                    f"{factor['mensaje']}"
                )

        st.warning(
            result["advertencia"]
        )
