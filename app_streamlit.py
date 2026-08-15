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
# Configuración visual de los campos
# ------------------------------------------------------------------

FIELD_CONFIG = {
    # ==============================================================
    # Datos del establecimiento educativo
    # ==============================================================

    "cole_area_ubicacion": {
        "section": "1. Información del establecimiento",
        "label": "¿En qué zona se ubica el establecimiento?",
        "help": (
            "Seleccione si el establecimiento se encuentra "
            "en una zona urbana o rural."
        ),
    },

    "cole_depto_ubicacion": {
        "section": "1. Información del establecimiento",
        "label": "¿En qué departamento está ubicado el establecimiento?",
        "help": (
            "Seleccione el departamento de "
            "la sede educativa."
        ),
    },

    "cole_jornada": {
        "section": "1. Información del establecimiento",
        "label": "¿Cuál es la jornada del establecimiento?",
        "help": (
            "Seleccione la jornada en la que estudia "
            "el estudiante."
        ),
    },

    "cole_naturaleza": {
        "section": "1. Información del establecimiento",
        "label": "¿Cuál es la naturaleza del establecimiento?",
        "help": (
            "Indique si el establecimiento es oficial "
            "o no oficial."
        ),
    },


    # ==============================================================
    # Hábitos y condiciones del estudiante
    # ==============================================================

    "estu_dedicacioninternet": {
        "section": "2. Hábitos y condiciones del estudiante",
        "label": "¿Cuánto tiempo dedica el estudiante a diario al uso de internet?",
        "help": (
            "Seleccione el tiempo diario de uso promedio "
            "de internet."
        ),
    },

    "estu_dedicacionlecturadiaria": {
        "section": "2. Hábitos y condiciones del estudiante",
        "label": "¿Cuánto tiempo dedica a la lectura por entretenimiento cada día?",
        "help": (
            "Indique el tiempo diario de lectura de libros, revistas u otros "
            "materiales como pasatiempo."
        ),
    },

    "estu_genero": {
        "section": "2. Hábitos y condiciones del estudiante",
        "label": "¿Cuál es el género del estudiante?",
        "help": (
            "Seleccione la opción disponible "
            "de género."
        ),
    },

    "estu_horassemanatrabaja": {
        "section": "2. Hábitos y condiciones del estudiante",
        "label": "¿Cuántas horas trabaja el estudiante a la semana?",
        "help": (
            "Seleccione el rango de horas "
            "trabajadas semanalmente."
        ),
    },

    "estu_tiporemuneracion": {
        "section": "2. Hábitos y condiciones del estudiante",
        "label": "Si trabaja, ¿qué tipo de remuneración recibe el estudiante por su trabajo?",
        "help": (
            "Indique si recibe pago en efectivo, en especie "
            "o mediante ambas modalidades."
        ),
    },


    # ==============================================================
    # Alimentación y condiciones del hogar
    # ==============================================================

    "fami_comecarnepescadohuevo": {
        "section": "3. Alimentación y condiciones del hogar",
        "label": "¿Con qué frecuencia el hogar consume carne, pescado o huevo?",
        "help": (
            "Seleccione la frecuencia semanal con que se consumen estos "
            "alimentos en el hogar."
        ),
    },

    "fami_comecerealfrutoslegumbre": {
        "section": "3. Alimentación y condiciones del hogar",
        "label": "¿Con qué frecuencia el hogar consume cereales, frutas o legumbres?",
        "help": (
            "Seleccione la frecuencia semanal con que se consumen estos "
            "alimentos en el hogar."
        ),
    },

    "fami_comelechederivados": {
        "section": "3. Alimentación y condiciones del hogar",
        "label": "¿Con qué frecuencia el hogar consume leche o sus derivados?",
        "help": (
            "Seleccione la frecuencia semanal con que se consumen estos "
            "alimentos en el hogar."
        ),
    },

    "fami_cuartoshogar": {
        "section": "3. Alimentación y condiciones del hogar",
        "label": "¿Cuántos cuartos tiene el hogar?",
        "help": (
            "Seleccione el número de cuartos "
            "de la vivienda."
        ),
    },

    "fami_personashogar": {
        "section": "3. Alimentación y condiciones del hogar",
        "label": "¿Cuántas personas conforman el hogar?",
        "help": (
            "Seleccione el número de personas que habitan en la vivienda, "
            "incluyendo al estudiante."
        ),
    },

    "fami_situacioneconomica": {
        "section": "3. Alimentación y condiciones del hogar",
        "label": "¿Cómo considera que ha cambiado la situación económica del hogar?",
        "help": (
            "Seleccione si la situación económica del hogar ha mejorado, "
            "empeorado o se mantiene igual frente al periodo anterior."
        ),
    },


    # ==============================================================
    # Educación y ocupación de los padres
    # ==============================================================

    "fami_educacionmadre": {
        "section": "4. Educación y ocupación de los padres",
        "label": "¿Cuál es el nivel educativo más alto de la madre?",
        "help": (
            "Seleccione el nivel educativo más alto "
            "de la madre."
        ),
    },

    "fami_educacionpadre": {
        "section": "4. Educación y ocupación de los padres",
        "label": "¿Cuál es el nivel educativo más alto del padre?",
        "help": (
            "Seleccione el nivel educativo más alto "
            "del padre."
        ),
    },

    "fami_trabajolabormadre": {
        "section": "4. Educación y ocupación de los padres",
        "label": "¿Cuál es la actividad laboral principal de la madre?",
        "help": (
            "Seleccione la actividad principal "
            "de ocupación de la madre."
        ),
    },

    "fami_trabajolaborpadre": {
        "section": "4. Educación y ocupación de los padres",
        "label": "¿Cuál es la actividad laboral principal del padre?",
        "help": (
            "Seleccione la actividad principal "
            "de ocupación del padre."
        ),
    },


    # ==============================================================
    # Condiciones socioeconómicas y recursos
    # ==============================================================

    "fami_estratovivienda": {
        "section": "5. Condiciones socioeconómicas y de recursos del hogar",
        "label": "¿Cuál es el estrato económico de la vivienda?",
        "help": (
            "Seleccione el estrato socioeconómico de la vivienda."
        ),
    },

    "fami_numlibros": {
        "section": "5. Condiciones socioeconómicas y de recursos del hogar",
        "label": "¿Cuántos libros hay aproximadamente en el hogar?",
        "help": (
            "Seleccione el número aproximado de libros disponibles "
            "en el hogar."
        ),
    },

    "fami_tieneautomovil": {
        "section": "5. Condiciones socioeconómicas y de recursos del hogar",
        "label": "¿El hogar dispone de automóvil?",
        "help": (
            "Seleccione si el hogar dispone "
            "de automóvil."
        ),
    },

    "fami_tienecomputador": {
        "section": "5. Condiciones socioeconómicas y de recursos del hogar",
        "label": "¿El hogar dispone de computador?",
        "help": (
            "Seleccione si el hogar dispone de computadores de escritorio "
            "o portátiles disponibles en el hogar."
        ),
    },

    "fami_tieneconsolavideojuegos": {
        "section": "5. Condiciones socioeconómicas y de recursos del hogar",
        "label": "¿El hogar dispone de consola de videojuegos?",
        "help": (
            "Seleccione si el hogar del estudiante "
            "dispone de consola de videojuegos."
        ),
    },

    "fami_tienehornomicroogas": {
        "section": "5. Condiciones socioeconómicas y de recursos del hogar",
        "label": "¿El hogar dispone de horno microondas?",
        "help": "Seleccione si el hogar del estudiante "
        "dispone de horno microondas.",
    },

    "fami_tieneinternet": {
        "section": "5. Condiciones socioeconómicas y de recursos del hogar",
        "label": "¿El hogar dispone de conexión a internet?",
        "help": (
            "Seleccione si la vivienda dispone habitualmente "
            "de conexión a internet."
        ),
    },

    "fami_tienelavadora": {
        "section": "5. Condiciones socioeconómicas y de recursos del hogar",
        "label": "¿El hogar dispone de lavadora?",
        "help": "Seleccione si el hogar del estudiante "
        "dispone de lavadora.",
    },

    "fami_tienemotocicleta": {
        "section": "5. Condiciones socioeconómicas y de recursos del hogar",
        "label": "¿El hogar dispone de motocicleta?",
        "help": (
            "Seleccione si el hogar del estudiante "
            "dispone de motocicleta."
        ),
    },

    "fami_tieneserviciotv": {
        "section": "5. Condiciones socioeconómicas y de recursos del hogar",
        "label": "¿El hogar dispone de servicio de televisión?",
        "help": (
            "Seleccione si dispone de televisión por cable, satélite "
            "o servicios equivalentes en el hogar del estudiante."
        ),
    },
}


# Validar que los textos de ayuda sean cadenas válidas para Streamlit.
for field_name, field_config in FIELD_CONFIG.items():
    help_value = field_config.get("help")

    if help_value is not None and not isinstance(help_value, str):
        raise TypeError(
            f"FIELD_CONFIG['{field_name}']['help'] debe ser texto, "
            f"pero se recibió {type(help_value).__name__}."
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
# Orden visual de las secciones
# ------------------------------------------------------------------

SECTION_ORDER = {
    "1. Información del establecimiento": 1,
    "2. Hábitos y condiciones del estudiante": 2,
    "3. Alimentación y condiciones del hogar": 3,
    "4. Educación y ocupación de los padres": 4,
    "5. Condiciones socioeconómicas y de recursos del hogar": 5,
    "Información adicional": 99,
}


# ------------------------------------------------------------------
# Formulario de predicción por pasos
# ------------------------------------------------------------------

SECTIONS = [
    section
    for section, _
    in sorted(
        SECTION_ORDER.items(),
        key=lambda item: item[1],
    )
    if section != "Información adicional"
]


if "paso_formulario" not in st.session_state:
    st.session_state["paso_formulario"] = 0


if "respuestas_formulario" not in st.session_state:
    st.session_state["respuestas_formulario"] = {}


total_steps = len(SECTIONS)

current_step = st.session_state[
    "paso_formulario"
]

current_section = SECTIONS[
    current_step
]


st.subheader("Formulario de evaluación")

st.write(
    f"**Paso {current_step + 1} de {total_steps}**"
)

st.progress(
    (current_step + 1) / total_steps
)

st.caption(
    current_section
)


current_features = [
    feature
    for feature in feature_order
    if FIELD_CONFIG.get(
        feature,
        {
            "section": "Información adicional",
        },
    )["section"] == current_section
]


if not current_features:
    st.error(
        "La sección actual no tiene variables configuradas."
    )
    st.stop()


with st.form(
    key=f"formulario_paso_{current_step}",
):

    st.markdown(
        f"### {current_section}"
    )

    st.info(
        "Complete los campos de esta sección antes "
        "de continuar."
    )

    for feature in current_features:

        config = FIELD_CONFIG.get(
            feature,
            {
                "section": "Información adicional",
                "label": feature,
                "help": None,
            },
        )

        label = config["label"]
        help_text = config.get("help")

        widget_key = f"input_{feature}"

        saved_value = st.session_state[
            "respuestas_formulario"
        ].get(
            feature
        )

        if feature in categorical_features:

            options = categories.get(
                feature,
                [],
            )

            if not options:
                st.error(
                    "No se encontraron categorías "
                    f"permitidas para '{feature}'."
                )
                st.stop()

            default_index = 0

            if (
                saved_value is not None
                and saved_value in options
            ):
                default_index = options.index(
                    saved_value
                )

            st.selectbox(
                label=label,
                options=options,
                index=default_index,
                help=help_text,
                key=widget_key,
            )

        else:

            numeric_default = (
                float(saved_value)
                if saved_value is not None
                else 0.0
            )

            st.number_input(
                label=label,
                value=numeric_default,
                help=help_text,
                key=widget_key,
            )

    left_column, right_column = st.columns(
        2
    )

    with left_column:

        previous_clicked = st.form_submit_button(
            "Anterior",
            use_container_width=True,
            disabled=current_step == 0,
        )

    with right_column:

        if current_step < total_steps - 1:

            next_clicked = st.form_submit_button(
                "Siguiente",
                type="primary",
                use_container_width=True,
            )

            evaluate_clicked = False

        else:

            next_clicked = False

            evaluate_clicked = st.form_submit_button(
                "Evaluar estudiante",
                type="primary",
                use_container_width=True,
            )


def guardar_respuestas_del_paso(
    features: list[str],
) -> None:
    """
    Guarda las respuestas del paso actual en un
    diccionario independiente de los widgets.
    """

    for feature in features:

        widget_key = f"input_{feature}"

        if widget_key in st.session_state:

            st.session_state[
                "respuestas_formulario"
            ][feature] = st.session_state[
                widget_key
            ]


if previous_clicked:

    guardar_respuestas_del_paso(
        current_features
    )

    st.session_state[
        "paso_formulario"
    ] = max(
        current_step - 1,
        0,
    )

    st.rerun()


if next_clicked:

    guardar_respuestas_del_paso(
        current_features
    )

    st.session_state[
        "paso_formulario"
    ] = min(
        current_step + 1,
        total_steps - 1,
    )

    st.rerun()


payload: dict[str, object] = {}
submitted = False


if evaluate_clicked:

    guardar_respuestas_del_paso(
        current_features
    )

    respuestas_guardadas = st.session_state[
        "respuestas_formulario"
    ]

    missing_features = [
        feature
        for feature in feature_order
        if feature not in respuestas_guardadas
    ]

    if missing_features:

        st.error(
            "Faltan campos por diligenciar. "
            "Regrese a las secciones anteriores."
        )

        st.code(
            str(missing_features)
        )

        st.stop()

    payload = {
        feature: respuestas_guardadas[
            feature
        ]
        for feature in feature_order
    }

    submitted = True
    
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


    # --------------------------------------------------------------
    # Aspectos favorables para casos sin riesgo
    # --------------------------------------------------------------

    factores_favorables = result.get(
        "factores_favorables",
        [],
    )


    if not result.get(
        "riesgo",
        False,
    ):

        if factores_favorables:

            st.markdown(
                "### Aspectos que contribuyeron favorablemente"
            )

            for factor in factores_favorables:

                dimension = factor.get(
                    "dimension",
                    "Aspecto favorable",
                )

                mensaje = factor.get(
                    "mensaje",
                    (
                        "Este aspecto contribuyó "
                        "favorablemente al resultado."
                    ),
                )

                st.markdown(
                    f"**{dimension}**  \n"
                    f"{mensaje}"
                )

        else:

            st.info(
                "No se identificaron aspectos favorables "
                "visibles suficientes para presentar una "
                "explicación individual."
            )


        st.markdown(
            "### Recomendaciones de mantenimiento"
        )

        st.markdown(
            "- Mantener los hábitos y condiciones educativas "
            "favorables identificadas.\n"
            "- Continuar con el seguimiento académico periódico.\n"
            "- Realizar una nueva evaluación si cambian las "
            "condiciones del estudiante o del hogar."
        )

        st.info(
            "La clasificación sin riesgo no garantiza la "
            "ausencia de futuras dificultades académicas."
        )


    # --------------------------------------------------------------
    # Advertencia general
    # --------------------------------------------------------------

    st.warning(
        result["advertencia"]
    )
