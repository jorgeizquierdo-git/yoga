import json
from utils import (NO_EVALUABLE, OK, FALLO, FACTOR_CAIDA, UMBRAL_APROBADO, _angulos_relevantes, _estado,
comprobar_encuadre, puntos_necesarios, extraer_puntos, medir_angulos)

def cargar_postura(ruta):
    with open(ruta, "r", encoding="utf-8") as f:
        datos = json.load(f)
    pose = datos["postura"]
    return pose

limite = lambda tolerancia: tolerancia * FACTOR_CAIDA
nota_angulo = lambda error, tolerancia: 1.0 if error <= tolerancia else (0 if error >= limite(tolerancia) else (limite(tolerancia) - error) / (limite(tolerancia) - tolerancia))
# Vaya desastre he hecho en este lambda xdxdxdxdxxdxd

def comparar_posturas(observados, esperado):
    detalle = {}

    for clave, datos in _angulos_relevantes(esperado):
        objetivo, tolerancia, observado = datos["objetivo"], datos["tol"], observados.get(clave)
        if observado is None:
            error = None

        detalle[clave] = {
            "observado": observado,
            "objetivo": objetivo,
            "tol": tolerancia,
            "error": error,
            "nota": None if not observado else nota_angulo(error, tolerancia),
            "estado": _estado(observado, error, tolerancia)
        }

    resultado = {
        "estado": None, 
        "puntuacion": None, 
        "total": len(detalle), 
        "avisos": [], 
        "detalle": detalle,
        "aciertos": sum(d["estado"] == OK for d in detalle.values())
    }

    sin_medir = [clave for clave, datos in detalle.items() if detalle["estado"] == NO_EVALUABLE]
    if sin_medir:
        resultado["estado"] = NO_EVALUABLE
        resultado["avisos"] = ["No se ven bien: " + ", ".join(c.replace("_", " ") for c in sin_medir)]
        return resultado

    resultado["puntuacion"] = 100.0 * sum(datos["nota"] for datos in detalle.values()) / len(detalle)
    resultado["estado"] = OK if resultado["puntuacion"] >= UMBRAL_APROBADO else FALLO


def comparar_pose(result, ruta):
    pose = cargar_postura(ruta)

    encuadre, avisos = comprobar_encuadre(result, puntos_necesarios(pose))
    puntos, visibilidad = extraer_puntos(result) if encuadre else (None, None)
    if puntos is None:
        return {
            "estado": NO_EVALUABLE, 
            "puntuacion": None, 
            "aciertos": 0,
            "total": len(_angulos_relevantes(pose)),
            "detalle": {},
            "avisos": avisos or ["No hay puntos disponibles"]
        }

    medidos = medir_angulos(puntos, pose, visibilidad)
    return comparar_posturas(medidos, pose)