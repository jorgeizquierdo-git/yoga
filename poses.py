import copy
import json
from utils import (NO_EVALUABLE, OK, FALLO, ERROR_MAXIMO, UMBRAL_APROBADO, NOMBRES, IDX, _angulos_relevantes,
comprobar_encuadre, puntos_necesarios, extraer_puntos, medir_angulos)

def cargar_postura(ruta):
    with open(ruta, "r", encoding="utf-8") as f:
        datos = json.load(f)
    pose = datos["postura"]
    return pose

# ---------------------------------------------------------------------------
# Simetría: la misma pose hecha con el otro lado del cuerpo
# ---------------------------------------------------------------------------
def _lado_opuesto(nombre):
    """"rodilla_izq" -> "rodilla_der" (y al revés). Lo que no tiene lado ("nariz") se queda igual."""
    if nombre.endswith("_izq"):
        return nombre[:-4] + "_der"
    if nombre.endswith("_der"):
        return nombre[:-4] + "_izq"
    # puntos de la cara: "ojo_izq_int", "ojo_der_ext"...
    if "_izq_" in nombre:
        return nombre.replace("_izq_", "_der_")
    if "_der_" in nombre:
        return nombre.replace("_der_", "_izq_")
    return nombre


def _punto_opuesto(p):
    """Acepta nombre ("rodilla_der") o índice (26) y devuelve el del otro lado en el mismo formato."""
    if isinstance(p, str):
        return _lado_opuesto(p)
    return IDX[_lado_opuesto(NOMBRES[int(p)])]


def espejar_pose(pose):
    """
    Devuelve una copia de la pose con izquierda y derecha intercambiadas, tanto en las claves de los
    ángulos ("rodilla_der" -> "rodilla_izq") como en sus puntos. Los objetivos y tolerancias no cambian:
    el ángulo es el mismo, solo cambia de qué lado del cuerpo es.
    """
    espejo = copy.deepcopy(pose)
    espejo["angulos"] = {
        _lado_opuesto(clave): {**datos, "puntos": [_punto_opuesto(p) for p in datos["puntos"]]}
        for clave, datos in pose["angulos"].items()
    }
    return espejo


def nota_angulo(error):
    return max(0.0, 1.0 - abs(error) / ERROR_MAXIMO)

def comparar_posturas(observados, esperado):
    detalle = {}

    for clave, datos in _angulos_relevantes(esperado):
        objetivo, observado = datos["objetivo"], observados.get(clave)
        error = None if observado is None else observado - objetivo

        detalle[clave] = {
            "observado": observado,
            "objetivo": objetivo,
            "tol": datos.get("tol", datos.get("tolerancia")),
            "error": error,
            "nota": None if error is None else nota_angulo(error)
        }

    resultado = {
        "estado": None,
        "puntuacion": None,
        "avisos": [],
        "detalle": detalle
    }

    sin_medir = [clave for clave, datos in detalle.items() if datos["nota"] is None]
    if sin_medir:
        resultado["estado"] = NO_EVALUABLE
        resultado["avisos"] = ["No se ven bien: " + ", ".join(c.replace("_", " ") for c in sin_medir)]
        return resultado

    resultado["puntuacion"] = 100.0 * sum(datos["nota"] for datos in detalle.values()) / len(detalle)
    resultado["estado"] = OK if resultado["puntuacion"] >= UMBRAL_APROBADO else FALLO
    return resultado


def _comparar_una_version(result, pose, ancho, alto):
    encuadre, avisos = comprobar_encuadre(result, puntos_necesarios(pose))
    puntos, visibilidad = extraer_puntos(result, ancho, alto) if encuadre else (None, None)
    if puntos is None:
        # Sin encuadre no se mide nada, pero se devuelven los ángulos de la pose (sin valor) para que
        # quien dibuje sepa qué puntos marcar como no medidos.
        resultado = comparar_posturas({}, pose)
        resultado["avisos"] = avisos or ["No hay puntos disponibles"]
        return resultado

    medidos = medir_angulos(puntos, pose, visibilidad)
    return comparar_posturas(medidos, pose)


def comparar_pose(result, pose_o_ruta, ancho, alto):
    """
    Compara el esqueleto con la pose tal cual y con su versión espejo, y se queda con la que mejor puntúa.
    Así da igual con qué lado hagas una pose asimétrica.

    ancho, alto: tamaño de la imagen que se pasó a MediaPipe (los ángulos se miden en píxeles).
    pose_o_ruta: ruta al .json o el diccionario de la postura ya cargado (mejor esto último si se llama
    en cada frame, para no leer el archivo 30 veces por segundo).

    El resultado incluye "lado": "original" o "espejo", y su "detalle" usa las claves de esa versión
    (p. ej. si gana el espejo, aparecerá "rodilla_izq" donde el .json decía "rodilla_der").
    """
    pose = pose_o_ruta if isinstance(pose_o_ruta, dict) else cargar_postura(pose_o_ruta)

    original = _comparar_una_version(result, pose, ancho, alto)
    espejo = _comparar_una_version(result, espejar_pose(pose), ancho, alto)
    original["lado"], espejo["lado"] = "original", "espejo"

    # Si una versión no se puede evaluar, gana la otra; si ninguna, se devuelve la original (con sus avisos).
    if espejo["puntuacion"] is None:
        return original
    if original["puntuacion"] is None:
        return espejo
    return espejo if espejo["puntuacion"] > original["puntuacion"] else original