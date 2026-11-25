import copy
import json
from utils import (
    ERROR_MAXIMO, UMBRAL_APROBADO, NOMBRES, IDX, PESO_POR_DEFECTO, COBERTURA_MINIMA, angulos_relevantes,
    comprobar_encuadre, puntos_necesarios, extraer_puntos, medir_angulos, puntos_fuera_de_imagen, indice
)


def cargar_postura(ruta):
    """
    Carga la postura de la ruta obtenida
    """
    with open(ruta, "r", encoding="utf-8") as f:
        return json.load(f)["postura"]


def lado_opuesto(nombre):
    """
    Intercambia los lados. Por ejemplo, rodilla_derecha pasa a ser rodilla_izquierda y vice versa.
    """
    if nombre.endswith("_izq"):
        return nombre[:-4] + "_der"

    if nombre.endswith("_der"):
        return nombre[:-4] + "_izq"

    if "_izq_" in nombre: # para los puntos de la cara, aunque no sean útiles la verdad
        return nombre.replace("_izq_", "_der_")

    if "_der_" in nombre:
        return nombre.replace("_der_", "_izq_")

    return nombre


def punto_opuesto(punto):
    """
    Igual que lado_opuesto pero acepta también el índice del punto
    """
    if isinstance(punto, str):
        return lado_opuesto(punto)

    return IDX[lado_opuesto(NOMBRES[int(punto)])]


def espejar_pose(pose):
    """
    Copia de la pose con la izquierda y la derecha intercambiadas. Los objetivos no cambian, solo de que lado son.
    Sirve para evaluar las posiciones independientemente de la direccion que tome el usuario.
    """
    espejo = copy.deepcopy(pose)
    espejo["angulos"] = {
        lado_opuesto(clave): {**datos, "puntos": [punto_opuesto(punto) for punto in datos["puntos"]]}
        for clave, datos in pose["angulos"].items()
    }
    return espejo


def nota_angulo(error):
    """
    Ecuación de cálculo de error. La nota es una ecuación lineal porque no queríamos complicarnos muhco
    """
    return max(0.0, 1.0 - abs(error) / ERROR_MAXIMO)


def comparar_posturas(observados, esperado):
    """
    Media ponderada de las notas de los ángulos que se han podido medir. Si lo medido no llega a COBERTURA_MINIMA
    del peso total, la postura pasa a ser 'no_evaluable'
    """
    detalle = {}
    for clave, datos in angulos_relevantes(esperado):
        objetivo, observado = datos["objetivo"], observados.get(clave)
        error = None if observado is None else observado - objetivo

        detalle[clave] = {
            "observado": observado,
            "objetivo": objetivo,
            "tol": datos.get("tol"),
            "peso": float(datos.get("peso", PESO_POR_DEFECTO)),
            "error": error,
            "nota": None if error is None else nota_angulo(error)
        }

    resultado = {
        "estado": None,
        "puntuacion": None,
        "cobertura": 0.0,
        "avisos": [],
        "detalle": detalle
    }

    peso_total = sum(dato["peso"] for dato in detalle.values())
    medidos = [d for d in detalle.values() if d["nota"] is not None]
    peso_medido = sum(dato["peso"] for dato in medidos)
    resultado["cobertura"] = peso_medido / peso_total if peso_total > 0 else 0.0

    sin_medir = [clave for clave, dato in detalle.items() if dato["nota"] is None]
    if sin_medir:
        resultado["avisos"].append("No se ven bien: " + ", ".join(clave.replace("_", " ") for clave in sin_medir))

    if peso_medido <= 0 or resultado["cobertura"] < COBERTURA_MINIMA: # Si todos los pesos calculados del esqueleto no llegan al 60% (por defecto) o no se midió nada
        resultado["estado"] = "no_evaluable"
        return resultado

    resultado["puntuacion"] = 100.0 * sum(dato["peso"] * dato["nota"] for dato in medidos) / peso_medido # Ponderación al resultado
    resultado["estado"] = "ok" if resultado["puntuacion"] >= UMBRAL_APROBADO else "fallo"
    
    return resultado


def comparar_una_version(result, pose, ancho, alto):
    """
    Mide los ángulos y los compara. Un punto fuera de la imagen o poco visible solo anula los ángulos que lo usan
    """
    puntos, visibilidad = extraer_puntos(result, ancho, alto)
    if puntos is None: # se devuelven los ángulos sin valor para que se pinten como no medidos
        resultado = comparar_posturas({}, pose)
        resultado["avisos"] = ["No se detecta ninguna persona"]
        return resultado

    medidos = medir_angulos(puntos, pose, visibilidad)
    fuera = puntos_fuera_de_imagen(result)

    for clave, datos in angulos_relevantes(pose):
        if any(indice(punto) in fuera for punto in datos["puntos"]): # el modelo se lo ha inventado
            medidos[clave] = None

    resultado = comparar_posturas(medidos, pose)
    _, avisos_encuadre = comprobar_encuadre(result, puntos_necesarios(pose))
    resultado["avisos"] = avisos_encuadre + resultado["avisos"]
    
    return resultado


def comparar_pose(result, pose, ancho, alto):
    """
    Compara con la pose y con su espejo y se queda con la que mejor puntúa, así da igual con qué lado la hagas.
    En "lado" indica cuál ha ganado. Perfecto para usuarios como yo que se despistan con la posición.
    """
    original = comparar_una_version(result, pose, ancho, alto)
    espejo = comparar_una_version(result, espejar_pose(pose), ancho, alto)
    original["lado"], espejo["lado"] = "original", "espejo"

    if espejo["puntuacion"] is None:
        return original

    if original["puntuacion"] is None:
        return espejo

    return espejo if espejo["puntuacion"] > original["puntuacion"] else original