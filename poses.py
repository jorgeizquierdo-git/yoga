import json
from utils import (NO_EVALUABLE, OK, FALLO, ERROR_MAXIMO, UMBRAL_APROBADO, _angulos_relevantes,
comprobar_encuadre, puntos_necesarios, extraer_puntos, medir_angulos)

def cargar_postura(ruta):
    with open(ruta, "r", encoding="utf-8") as f:
        datos = json.load(f)
    pose = datos["postura"]
    return pose

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


def comparar_pose(result, ruta):
    pose = cargar_postura(ruta)

    encuadre, avisos = comprobar_encuadre(result, puntos_necesarios(pose))
    puntos, visibilidad = extraer_puntos(result) if encuadre else (None, None)
    if puntos is None:
        return {
            "estado": NO_EVALUABLE,
            "puntuacion": None,
            "avisos": avisos or ["No hay puntos disponibles"],
            "detalle": {}
        }

    medidos = medir_angulos(puntos, pose, visibilidad)
    return comparar_posturas(medidos, pose)