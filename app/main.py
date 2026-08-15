from __future__ import annotations

import json
from pathlib import Path

import joblib
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException

import numpy as np
import xgboost as xgb

# Carpeta principal del proyecto.
BASE_DIR = Path(__file__).resolve().parent.parent

# Carpeta donde están los artefactos del modelo.
ARTIFACT_DIR = BASE_DIR / "artifacts"

MODEL_PATH = (
    ARTIFACT_DIR
    / "s0_xgboost_pipeline.joblib"
)

METADATA_PATH = (
    ARTIFACT_DIR
    / "model_metadata.json"
)

SCHEMA_PATH = (
    ARTIFACT_DIR
    / "input_schema.json"
)

CATEGORIES_PATH = (
    ARTIFACT_DIR
    / "encoder_categories.json"
)

FEATURE_MAP_PATH = (
    ARTIFACT_DIR
    / "transformed_feature_map.json"
)

POLICY_PATH = (
    ARTIFACT_DIR
    / "explanation_policy.json"
)

def cargar_json(path: Path) -> dict:
    """Carga un archivo JSON con codificación UTF-8."""

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


# Validar existencia antes de iniciar la API.
for required_path in [
    MODEL_PATH,
    METADATA_PATH,
    SCHEMA_PATH,
    CATEGORIES_PATH,
    FEATURE_MAP_PATH,
    POLICY_PATH,
]:
    if not required_path.exists():
        raise FileNotFoundError(
            f"No se encontró el artefacto: {required_path}"
        )
    
# Cargar artefactos una sola vez al iniciar.
pipeline = joblib.load(
    MODEL_PATH
)

metadata = cargar_json(
    METADATA_PATH
)

schema = cargar_json(
    SCHEMA_PATH
)

categories = cargar_json(
    CATEGORIES_PATH
)

feature_map = cargar_json(
    FEATURE_MAP_PATH
)

FEATURE_LOOKUP = {
    item["transformed_feature"]: item
    for item in feature_map.get("features", [])
    if item.get("transformed_feature")
    and item.get("original_feature")
}
policy = cargar_json(
    POLICY_PATH
)

def calcular_contribuciones_locales(
    input_frame: pd.DataFrame,
) -> tuple[list[str], np.ndarray]:
    """
    Calcula contribuciones Tree SHAP nativas de XGBoost.

    Devuelve:
    - nombres de variables transformadas;
    - contribuciones locales, sin incluir el término base.
    """

    preprocessor = pipeline.named_steps[
        "preprocess"
    ]

    model = pipeline.named_steps[
        "model"
    ]

    booster = model.get_booster()

    transformed_row = preprocessor.transform(
        input_frame
    )

    if hasattr(
        transformed_row,
        "toarray",
    ):
        transformed_row = (
            transformed_row.toarray()
        )

    transformed_row = np.asarray(
        transformed_row,
        dtype=np.float32,
    )

    transformed_names = list(
        preprocessor.get_feature_names_out()
    )

    if transformed_row.shape[1] != len(
        transformed_names
    ):
        raise RuntimeError(
            "No coincide el número de columnas "
            "transformadas con sus nombres."
        )

    dmatrix = xgb.DMatrix(
        transformed_row
    )

    contributions = booster.predict(
        dmatrix,
        pred_contribs=True,
        validate_features=False,
    )

    contributions = np.asarray(
        contributions,
        dtype=float,
    )

    if contributions.ndim != 2:
        raise RuntimeError(
            "Formato inesperado de contribuciones: "
            f"{contributions.shape}"
        )

    expected_columns = (
        len(transformed_names) + 1
    )

    if contributions.shape != (
        1,
        expected_columns,
    ):
        raise RuntimeError(
            "La salida de contribuciones no coincide "
            "con el número de variables transformadas."
        )

    local_values = contributions[
        0,
        :-1,
    ]

    return (
        transformed_names,
        local_values,
    )

from collections import defaultdict


MENSAJES_AMIGABLES = {
    "estu_dedicacioninternet": {
        "dimension": "Uso del tiempo y conectividad",
        "mensaje": (
            "Conviene revisar las condiciones de acceso y uso "
            "de internet para apoyar las actividades académicas."
        ),
    },
    "estu_dedicacionlecturadiaria": {
        "dimension": "Hábitos académicos",
        "mensaje": (
            "Conviene fortalecer una rutina regular de lectura "
            "y aprendizaje autónomo."
        ),
    },
    "estu_horassemanatrabaja": {
        "dimension": "Disponibilidad de tiempo",
        "mensaje": (
            "La disponibilidad de tiempo para el estudio podría "
            "requerir organización y acompañamiento adicional."
        ),
    },
    "estu_tiporemuneracion": {
        "dimension": "Condiciones de dedicación del estudiante",
        "mensaje": (
            "Algunas condiciones relacionadas con la dedicación "
            "del estudiante podrían requerir acompañamiento."
        ),
    },
    "fami_situacioneconomica": {
        "dimension": "Estabilidad económica del hogar",
        "mensaje": (
            "La situación económica reciente del hogar podría hacer "
            "conveniente un apoyo educativo complementario."
        ),
    },
    "fami_estratovivienda": {
        "dimension": "Condiciones socioeconómicas del hogar",
        "mensaje": (
            "Algunas condiciones socioeconómicas del hogar podrían "
            "requerir apoyos educativos complementarios."
        ),
    },
    "fami_personashogar": {
        "dimension": "Condiciones para el estudio en el hogar",
        "mensaje": (
            "Algunas condiciones de convivencia podrían requerir "
            "apoyos para facilitar el estudio."
        ),
    },
    "fami_cuartoshogar": {
        "dimension": "Condiciones para el estudio en el hogar",
        "mensaje": (
            "Algunas condiciones del hogar podrían requerir apoyos "
            "para facilitar el tiempo y el espacio de estudio."
        ),
    },
    "fami_educacionmadre": {
        "dimension": "Acompañamiento académico familiar",
        "mensaje": (
            "Puede ser útil fortalecer las redes de orientación "
            "y acompañamiento académico familiar."
        ),
    },
    "fami_educacionpadre": {
        "dimension": "Acompañamiento académico familiar",
        "mensaje": (
            "Puede ser útil fortalecer las redes de orientación "
            "y acompañamiento académico familiar."
        ),
    },
    "fami_numlibros": {
        "dimension": "Recursos culturales y de lectura",
        "mensaje": (
            "Puede ser beneficioso ampliar el acceso a materiales "
            "de lectura y consulta."
        ),
    },
    "fami_tienecomputador": {
        "dimension": "Recursos para el aprendizaje",
        "mensaje": (
            "Se identifican oportunidades para fortalecer el acceso "
            "a recursos de estudio."
        ),
    },
    "fami_tieneinternet": {
        "dimension": "Conectividad para el aprendizaje",
        "mensaje": (
            "Conviene revisar alternativas de conectividad que apoyen "
            "las actividades académicas."
        ),
    },
    "fami_comecarnepescadohuevo": {
        "dimension": "Condiciones de bienestar del hogar",
        "mensaje": (
            "Algunas condiciones de bienestar del hogar podrían "
            "requerir acompañamiento complementario."
        ),
    },
    "fami_comecerealfrutoslegumbre": {
        "dimension": "Condiciones de bienestar del hogar",
        "mensaje": (
            "Algunas condiciones de bienestar del hogar podrían "
            "requerir acompañamiento complementario."
        ),
    },
    "fami_comelechederivados": {
        "dimension": "Condiciones de bienestar del hogar",
        "mensaje": (
            "Algunas condiciones de bienestar del hogar podrían "
            "requerir acompañamiento complementario."
        ),
    },
}

# ============================================================
# Mensajes para aspectos que reducen la estimación de riesgo
# ============================================================

MENSAJES_FAVORABLES = {
    "estu_dedicacioninternet": {
        "dimension": "Uso del tiempo y conectividad",
        "mensaje": (
            "Las condiciones registradas de acceso y uso "
            "de internet contribuyeron favorablemente al resultado."
        ),
    },

    "estu_dedicacionlecturadiaria": {
        "dimension": "Hábitos académicos",
        "mensaje": (
            "Los hábitos de lectura registrados contribuyeron "
            "favorablemente al resultado."
        ),
    },

    "estu_horassemanatrabaja": {
        "dimension": "Disponibilidad de tiempo",
        "mensaje": (
            "La disponibilidad de tiempo registrada contribuyó "
            "favorablemente al resultado."
        ),
    },

    "fami_situacioneconomica": {
        "dimension": "Estabilidad económica del hogar",
        "mensaje": (
            "La situación económica registrada contribuyó "
            "favorablemente al resultado."
        ),
    },

    "fami_educacionmadre": {
        "dimension": "Acompañamiento académico familiar",
        "mensaje": (
            "Las condiciones registradas de acompañamiento familiar "
            "contribuyeron favorablemente al resultado."
        ),
    },

    "fami_educacionpadre": {
        "dimension": "Acompañamiento académico familiar",
        "mensaje": (
            "Las condiciones registradas de acompañamiento familiar "
            "contribuyeron favorablemente al resultado."
        ),
    },

    "fami_numlibros": {
        "dimension": "Recursos culturales y de lectura",
        "mensaje": (
            "La disponibilidad registrada de materiales de lectura "
            "contribuyó favorablemente al resultado."
        ),
    },

    "fami_tienecomputador": {
        "dimension": "Recursos para el aprendizaje",
        "mensaje": (
            "La disponibilidad registrada de recursos tecnológicos "
            "contribuyó favorablemente al resultado."
        ),
    },

    "fami_tieneinternet": {
        "dimension": "Conectividad para el aprendizaje",
        "mensaje": (
            "Las condiciones registradas de conectividad contribuyeron "
            "favorablemente al resultado."
        ),
    },

    "fami_personashogar": {
        "dimension": "Condiciones para el estudio en el hogar",
        "mensaje": (
            "Las condiciones registradas del hogar contribuyeron "
            "favorablemente al resultado."
        ),
    },
}

def obtener_factores_visibles(
    input_frame: pd.DataFrame,
    max_factores: int = 3,
) -> list[dict[str, str]]:
    """
    Selecciona factores individuales que aumentaron el riesgo.

    No devuelve:
    - nombres técnicos;
    - valores originales;
    - variables institucionales;
    - variables territoriales;
    - contribuciones numéricas.
    """

    transformed_names, local_values = (
        calcular_contribuciones_locales(
            input_frame
        )
    )

    blocked_prefixes = tuple(
        policy.get(
            "blocked_prefixes",
            [],
        )
    )

    # Compatibilidad con explanation_policy.json v2.0.0
    # y con versiones anteriores.
    blocked_features = set(
        policy.get(
            "exclude_features",
            policy.get(
                "blocked_features",
                [],
            ),
        )
    )

    # Lista segura: solo variables con mensaje amigable.
    allowed_features = set(
        MENSAJES_AMIGABLES
    )

    # Si la política define explícitamente variables visibles,
    # se aplica como restricción adicional.
    configured_allowed = policy.get(
        "allowed_visible_features"
    )

    if configured_allowed:
        allowed_features &= set(
            configured_allowed
        )

    grouped = defaultdict(float)

    for transformed_name, contribution in zip(
        transformed_names,
        local_values,
        strict=True,
    ):
        contribution = float(
            contribution
        )

        # Solo factores que empujan hacia riesgo.
        if contribution <= 0:
            continue

        mapping = FEATURE_LOOKUP.get(transformed_name)

        if mapping is None:
            continue

        original_feature = mapping.get(
            "original_feature"
        )

        if not original_feature:
            continue

        if not mapping.get(
            "show_to_user",
            False,
        ):
            continue

        if mapping.get(
            "sensitive",
            False,
        ):
            continue

        if original_feature in blocked_features:
            continue

        if (
            blocked_prefixes
            and original_feature.startswith(
                blocked_prefixes
            )
        ):
            continue

        if original_feature not in allowed_features:
            continue

        message_config = MENSAJES_AMIGABLES.get(
            original_feature
        )

        if message_config is None:
            continue

        dimension = message_config[
            "dimension"
        ]

        grouped[dimension] += contribution

    ordered_dimensions = sorted(
        grouped.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:max_factores]

    factores = []

    for dimension, _ in ordered_dimensions:
        matching_config = next(
            config
            for config
            in MENSAJES_AMIGABLES.values()
            if config["dimension"] == dimension
        )

        factores.append(
            {
                "dimension": dimension,
                "mensaje": matching_config[
                    "mensaje"
                ],
            }
        )

    return factores

def obtener_factores_favorables(
    input_frame: pd.DataFrame,
    max_factores: int = 3,
) -> list[dict[str, str]]:
    """
    Selecciona aspectos que redujeron la estimación local
    de riesgo para el registro evaluado.

    No devuelve nombres técnicos, valores originales
    ni contribuciones numéricas al usuario.

    Incluye trazabilidad en logs para diagnosticar
    por qué ciertos factores favorables son descartados.
    """

    transformed_names, local_values = (
        calcular_contribuciones_locales(
            input_frame
        )
    )

    allowed_features = set(
        MENSAJES_FAVORABLES
    )

    blocked_features = set(
        policy.get(
            "exclude_features",
            policy.get(
                "blocked_features",
                [],
            ),
        )
    )

    blocked_prefixes = tuple(
        policy.get(
            "blocked_prefixes",
            [],
        )
    )

    grouped = defaultdict(float)

    # ----------------------------------------------------------
    # Contadores de diagnóstico
    # ----------------------------------------------------------

    diagnostico = {
        "total_transformadas": 0,
        "contribuciones_negativas": 0,
        "sin_mapping": 0,
        "sin_original_feature": 0,
        "no_visible": 0,
        "sensible": 0,
        "bloqueada": 0,
        "prefijo_bloqueado": 0,
        "no_autorizada": 0,
        "sin_mensaje": 0,
        "aceptadas": 0,
    }


    for transformed_name, contribution in zip(
        transformed_names,
        local_values,
        strict=True,
    ):

        diagnostico["total_transformadas"] += 1

        contribution = float(
            contribution
        )

        # Solo contribuciones que reducen
        # la estimación de riesgo.
        if contribution >= 0:
            continue

        diagnostico[
            "contribuciones_negativas"
        ] += 1


        mapping = FEATURE_LOOKUP.get(
            transformed_name
        )

        if mapping is None:
            diagnostico[
                "sin_mapping"
            ] += 1
            continue


        original_feature = mapping.get(
            "original_feature"
        )

        if not original_feature:
            diagnostico[
                "sin_original_feature"
            ] += 1
            continue


        if not mapping.get(
            "show_to_user",
            False,
        ):
            diagnostico[
                "no_visible"
            ] += 1
            continue


        if mapping.get(
            "sensitive",
            False,
        ):
            diagnostico[
                "sensible"
            ] += 1
            continue


        if original_feature in blocked_features:
            diagnostico[
                "bloqueada"
            ] += 1
            continue


        if (
            blocked_prefixes
            and original_feature.startswith(
                blocked_prefixes
            )
        ):
            diagnostico[
                "prefijo_bloqueado"
            ] += 1
            continue


        if original_feature not in allowed_features:
            diagnostico[
                "no_autorizada"
            ] += 1
            continue


        message_config = MENSAJES_FAVORABLES.get(
            original_feature
        )

        if message_config is None:
            diagnostico[
                "sin_mensaje"
            ] += 1
            continue


        dimension = message_config[
            "dimension"
        ]

        grouped[dimension] += abs(
            contribution
        )

        diagnostico[
            "aceptadas"
        ] += 1


    # ----------------------------------------------------------
    # Ordenar dimensiones favorables por magnitud
    # ----------------------------------------------------------

    ordered_dimensions = sorted(
        grouped.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:max_factores]


    factores_favorables = []


    for dimension, _ in ordered_dimensions:

        matching_config = next(
            (
                config
                for config
                in MENSAJES_FAVORABLES.values()
                if config["dimension"] == dimension
            ),
            None,
        )

        if matching_config is None:
            continue


        factores_favorables.append(
            {
                "dimension": dimension,
                "mensaje": matching_config[
                    "mensaje"
                ],
            }
        )


    # ----------------------------------------------------------
    # Diagnóstico en logs de Render
    # ----------------------------------------------------------

    print(
        "DIAGNOSTICO_FACTORES_FAVORABLES:",
        diagnostico,
    )

    print(
        "DIMENSIONES_FAVORABLES_AGRUPADAS:",
        dict(grouped),
    )

    print(
        "FACTORES_FAVORABLES_FINALES:",
        factores_favorables,
    )


    return factores_favorables

app = FastAPI(
    title="SAT Riesgo Educativo",
    version="1.0.0",
)


@app.get("/health")
def health() -> dict:
    """Confirma que la API y el modelo están disponibles."""

    return {
        "status": "ok",
        "service": "SAT Riesgo Educativo",
        "model_loaded": True,
        "model_name": metadata.get(
            "model_name",
            "S0-XGBoost",
        ),
        "number_of_features": schema.get(
            "number_of_features",
        ),
        "threshold": metadata.get(
            "risk_threshold",
        ),
        "pipeline_steps": list(
            pipeline.named_steps.keys()
        ),
    }

@app.get("/schema")
def obtener_esquema() -> dict[str, Any]:
    """
    Devuelve las variables requeridas por el modelo,
    su tipología y las categorías permitidas para
    construir el formulario de entrada.
    """

    features = schema["features"]

    # Identificar las variables según el tipo declarado
    # en input_schema.json.
    numeric_features = [
        feature["name"]
        for feature in features
        if feature.get("type") == "numeric"
    ]

    categorical_features = [
        feature["name"]
        for feature in features
        if feature.get("type") == "categorical"
    ]

    # Construir el diccionario que Streamlit espera:
    #
    # {
    #     "nombre_variable": ["categoria_1", "categoria_2"]
    # }
    #
    # Las categorías se obtienen directamente de
    # allowed_values dentro de input_schema.json.
    categories_by_feature = {
        feature["name"]: feature.get(
            "allowed_values",
            [],
        )
        for feature in features
        if feature.get("type") == "categorical"
    }

    allow_null_values = any(
        feature.get("nullable", False)
        for feature in features
    )

    return {
        "schema_version": schema.get("schema_version"),
        "model_name": schema.get("model_name"),
        "number_of_features": schema["number_of_features"],
        "feature_order": schema["feature_order"],
        "features": features,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "categories": categories_by_feature,
        "allow_null_values": allow_null_values,
        "allow_extra_columns": False,
    }
@app.post("/predict")
def predecir(payload: dict[str, Any]) -> dict:
    """
    Genera la predicción de riesgo educativo.

    La entrada debe contener exactamente las variables
    definidas en input_schema.json.
    """

    feature_order = schema["feature_order"]

    expected_features = set(feature_order)
    received_features = set(payload)

    missing_features = sorted(
        expected_features
        - received_features
    )

    extra_features = sorted(
        received_features
        - expected_features
    )

    if missing_features:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "Faltan variables obligatorias",
                "variables": missing_features,
            },
        )

    if extra_features:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "Se recibieron variables no autorizadas",
                "variables": extra_features,
            },
        )

    input_frame = pd.DataFrame(
        [
            {
                feature: payload[feature]
                for feature in feature_order
            }
        ],
        columns=feature_order,
    )

    null_columns = (
        input_frame
        .columns[
            input_frame.isna().any()
        ]
        .tolist()
    )

    if null_columns:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "No se permiten valores nulos",
                "variables": null_columns,
            },
        )

    model = pipeline.named_steps["model"]
    classes = list(model.classes_)

    if 1 not in classes:
        raise HTTPException(
            status_code=500,
            detail=(
                "La clase positiva 1 no está "
                "configurada en el modelo."
            ),
        )

    positive_index = classes.index(1)

    try:
        probability = float(
            pipeline.predict_proba(
                input_frame
            )[0, positive_index]
        )

    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=(
                "Los valores recibidos no son compatibles "
                "con el esquema del modelo."
            ),
        ) from exc

    threshold = float(
        metadata["risk_threshold"]
    )

    is_risk = (
        probability >= threshold
    )

    maximum_factors = int(
        policy.get(
            "maximum_factors_returned",
            3,
        )
    )

    try:
        factores = (
            obtener_factores_visibles(
                input_frame=input_frame,
                max_factores=maximum_factors,
            )
            if is_risk
            else []
        )

    try:
        factores_favorables = (
            obtener_factores_favorables(
                input_frame=input_frame,
                max_factores=3,
            )
            if not is_risk
            else []
        )

except Exception as exc:

    print(
        "Error al generar factores favorables:",
        repr(exc),
    )

    factores_favorables = []

    except Exception as exc:
        # La predicción principal no debe fallar si falla
        # únicamente la capa de explicabilidad.
        print(
            "Error al generar factores explicativos:",
            repr(exc),
        )
        factores = []

    return {
    "resultado": (
        "Riesgo identificado"
        if is_risk
        else "Sin riesgo identificado"
    ),
    "riesgo": is_risk,
    "probabilidad_estimada": round(
        probability,
        6,
    ),
    "umbral": threshold,
    "modelo": metadata.get(
        "model_name",
        "S0-XGBoost",
    ),
    "factores": factores,
    "factores_favorables": factores_favorables,
    "advertencia": (
        "Resultado orientativo generado por un "
        "modelo estadístico. No constituye un diagnóstico."
    ),
    "mensaje_factores": policy.get(
        "user_message"
    ),
}
