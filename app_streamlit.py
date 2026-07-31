from __future__ import annotations

import hashlib
import hmac
import os

import requests
import streamlit as st


# ------------------------------------------------------------------
# Configuración general
# ------------------------------------------------------------------

# En Render, API_URL se obtiene desde una variable de entorno.
# En ejecución local, se utiliza FastAPI en el puerto 8000.
API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000",
).rstrip("/")


st.set_page_config(
    page_title="Sistema de Alerta Temprana",
    page_icon="📘",
    layout="centered",
)


# ------------------------------------------------------------------
# Control de acceso
# ------------------------------------------------------------------

def verificar_acceso() -> None:
    """
    Solicita y valida la contraseña antes de permitir
    el acceso al Sistema de Alerta Temprana.

    La contraseña correcta se obtiene desde la variable
    de entorno APP_PASSWORD.
    """

    password_correcta = os.getenv("APP_PASSWORD")

    if not password_correcta:
        st.error(
            "No se configuró la contraseña de acceso "
            "al sistema."
        )
        st.stop()

    # Si el usuario ya se autenticó durante esta sesión,
    # no se vuelve a solicitar la contraseña.
    if st.session_state.get("autenticado", False):
        return

    st.title("Sistema de Alerta Temprana")
    st.caption("Acceso restringido.")

    password_ingresada = st.text_input(
        "Contraseña",
        type="password",
        key="password_acceso",
    )

    if st.button(
        "Ingresar",
        type="primary",
    ):
        hash_ingresado = hashlib.sha256(
            password_ingresada.encode("utf-8")
        ).hexdigest()

        hash_correcto = hashlib.sha256(
            password_correcta.encode("utf-8")
        ).hexdigest()

        acceso_valido = hmac.compare_digest(
            hash_ingresado,
            hash_correcto,
        )

        if acceso_valido:
            st.session_state["autenticado"] = True
            st.rerun()

        st.error("Contraseña incorrecta.")

    # Impide que se ejecute el resto de la aplicación
    # mientras el usuario no esté autenticado.
    st.stop()


verificar_acceso()


# ------------------------------------------------------------------
# Encabezado de la aplicación
# ------------------------------------------------------------------

st.title("Sistema de Alerta Temprana")

st.caption(
    "Herramienta de apoyo para identificar posibles "
    "necesidades de acompañamiento educativo."
)


# ------------------------------------------------------------------
# Consulta del esquema de entrada
# ------------------------------------------------------------------

@st.cache_data(ttl=300)
def cargar_esquema() -> dict:
    """
    Consulta desde FastAPI las variables requeridas,
    su orden y las categorías permitidas por el modelo.
    """

    response = requests.get(
        f"{API_URL}/schema",
        timeout=30,
    )

    response.raise_for_status()

    return response.json()


try:
    schema = cargar_esquema()

except requests.Timeout:
    st.error(
        "La API tardó demasiado en responder. "
        "Es posible que el servicio esté iniciando."
    )
    st.stop()

except requests.ConnectionError:
    st.error(
        "No fue posible establecer conexión con la API."
    )
    st.stop()

except requests.HTTPError as exc:
    st.error(
        "La API respondió con un error al consultar "
        "el esquema de variables."
    )
    st.code(str(exc))
    st.stop()

except requests.RequestException as exc:
    st.error(
        "Ocurrió un error inesperado al consultar la API."
    )
    st.code(str(exc))
    st.stop()


# ------------------------------------------------------------------
# Validación de la respuesta del esquema
# ------------------------------------------------------------------

required_schema_keys = {
    "number_of_features",
    "feature_order",
    "categorical_features",
    "categories",
}

missing_schema_keys = required_schema_keys.difference(
    schema.keys()
)

if missing_schema_keys:
    st.error(
        "La respuesta del endpoint /schema está incompleta."
    )
    st.code(
        f"Claves faltantes: "
        f"{sorted(missing_schema_keys)}"
    )
    st.stop()


feature_order = schema["feature_order"]
categorical_features = set(
    schema["categorical_features"]
)
categories = schema["categories"]


if len(feature_order) != schema["number_of_features"]:
    st.error(
        "El número de variables declarado por la API "
        "no coincide con el orden de entrada."
    )
    st.stop()


st.success(
    f"Modelo disponible con "
    f"{schema['number_of_features']} variables."
)


# ------------------------------------------------------------------
# Formulario de predicción
# ------------------------------------------------------------------

payload: dict[str, object] = {}

with st.form("formulario_prediccion"):

    st.subheader("Información del estudiante")

    for feature in feature_order:

        # Variables categóricas:
        # solo se muestran las categorías aprendidas
        # durante el entrenamiento.
        if feature in categorical_features:

            options = categories.get(
                feature,
                [],
            )

            if not options:
                st.error(
                    f"No se encontraron categorías "
                    f"permitidas para la variable "
                    f"'{feature}'."
                )
                st.stop()

            payload[feature] = st.selectbox(
                label=feature,
                options=options,
                key=f"input_{feature}",
            )

        # Este bloque se utilizará únicamente si el
        # esquema contiene variables numéricas.
        else:
            payload[feature] = st.number_input(
                label=feature,
                value=0.0,
                key=f"input_{feature}",
            )

    submitted = st.form_submit_button(
        "Evaluar estudiante",
        type="primary",
        use_container_width=True,
    )


# ------------------------------------------------------------------
# Solicitud de predicción
# ------------------------------------------------------------------

if submitted:

    with st.spinner(
        "Analizando la información del estudiante..."
    ):

        try:
            response = requests.post(
                f"{API_URL}/predict",
                json=payload,
                timeout=60,
            )

            response.raise_for_status()
            result = response.json()

        except requests.Timeout:
            st.error(
                "La API tardó demasiado en generar "
                "la predicción."
            )
            st.stop()

        except requests.ConnectionError:
            st.error(
                "No fue posible conectarse con el "
                "servicio de predicción."
            )
            st.stop()

        except requests.HTTPError as exc:
            st.error(
                "La API rechazó la solicitud de predicción."
            )

            try:
                error_detail = response.json()
                st.json(error_detail)
            except ValueError:
                st.code(str(exc))

            st.stop()

        except requests.RequestException as exc:
            st.error(
                "Ocurrió un error inesperado durante "
                "la solicitud de predicción."
            )
            st.code(str(exc))
            st.stop()

        except ValueError:
            st.error(
                "La respuesta de la API no tiene "
                "un formato JSON válido."
            )
            st.stop()


    # --------------------------------------------------------------
    # Presentación del resultado
    # --------------------------------------------------------------

    required_result_keys = {
        "resultado",
        "probabilidad_estimada",
        "advertencia",
    }

    missing_result_keys = required_result_keys.difference(
        result.keys()
    )

    if missing_result_keys:
        st.error(
            "La respuesta de predicción está incompleta."
        )
        st.code(
            f"Claves faltantes: "
            f"{sorted(missing_result_keys)}"
        )
        st.stop()


    st.divider()

    st.subheader(
        result["resultado"]
    )

    st.metric(
        label="Probabilidad estimada",
        value=(
            f"{result['probabilidad_estimada']:.1%}"
        ),
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

            dimension = factor.get(
                "dimension",
                "Aspecto relevante",
            )

            mensaje = factor.get(
                "mensaje",
                "Sin descripción disponible.",
            )

            st.markdown(
                f"**{dimension}**  \n"
                f"{mensaje}"
            )


    st.warning(
        result["advertencia"]
    )
