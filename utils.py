"""
Archivo para funciones generales y de ayuda: configuración global y funciones matemáticas.

Flujo previsto en el juego:
    result = landmarker.detect_for_video(...)

    # 1) Antes de empezar (y durante el juego): ¿se ve el cuerpo entero?
    ok, avisos = comprobar_encuadre(result, puntos_necesarios(pose))

    # 2) Medir y evaluar la postura
    puntos, visib = extraer_puntos(result)              # (33, 3) y (33,)
    medidos = medir_angulos(puntos, pose, visib)        # {"rodilla_der": 170.2, "codo_izq": None, ...}
    estado, detalle = estado_postura(medidos, pose)     # "ok" | "fallo" | "no_evaluable"

donde `pose` es el diccionario de una postura del .json (el valor de la clave "postura"),
es decir, el que contiene "nombre", "mantener_s" y "angulos".

Estados de un ángulo o de una postura:
    OK            -> medido y dentro de la tolerancia
    FALLO         -> medido y fuera de la tolerancia (el jugador se equivoca)
    NO_EVALUABLE  -> no se pudo medir por poca visibilidad (el fallo es de la cámara, no del jugador)
En el juego, NO_EVALUABLE debe PAUSAR (o no contar el tiempo de "mantener_s") y pedir que se ajuste
el jugador, pero nunca penalizar.

Ángulos no relevantes: en el .json, un ángulo con "objetivo": null (así los deja registrador_poses.py
cuando no se marca como relevante) se IGNORA por completo: no se mide, no se evalúa y sus puntos no se
exigen en el encuadre. Un ángulo sin la clave "objetivo" (p. ej. los de depuración del tracker, que solo
se miden) sí se mide, pero no se puede evaluar.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Configuración global
# ---------------------------------------------------------------------------
# True  -> coordenadas "world" (wx, wy, wz) en metros: sin distorsión por el formato de la imagen.
# False -> coordenadas normalizadas (x, y, z): x e y van en unidades distintas (ancho/alto), así que
#          los ángulos 3D salen distorsionados. Solo para comparar.
USAR_WORLD = True

# Un punto se considera "fuera de la imagen" si queda a menos de este margen del borde (0-1).
MARGEN_ENCUADRE = 0.02

# Estados
OK = "ok"
FALLO = "fallo"
NO_EVALUABLE = "no_evaluable"

# Nombres de los 33 landmarks de MediaPipe Pose (índice de la lista = índice del landmark).
# "izq"/"der" son las del propio cuerpo de la persona, según el modelo.
NOMBRES = [
    "nariz",
    "ojo_izq_int", "ojo_izq", "ojo_izq_ext",
    "ojo_der_int", "ojo_der", "ojo_der_ext",
    "oreja_izq", "oreja_der",
    "boca_izq", "boca_der",
    "hombro_izq", "hombro_der",
    "codo_izq", "codo_der",
    "muneca_izq", "muneca_der",
    "meñique_izq", "meñique_der",
    "indice_izq", "indice_der",
    "pulgar_izq", "pulgar_der",
    "cadera_izq", "cadera_der",
    "rodilla_izq", "rodilla_der",
    "tobillo_izq", "tobillo_der",
    "talon_izq", "talon_der",
    "punta_pie_izq", "punta_pie_der",
]

ANGULOS = {
    "rodilla_izq": {"puntos": ["cadera_izq", "rodilla_izq", "tobillo_izq"]},
    "rodilla_der": {"puntos": ["cadera_der", "rodilla_der", "tobillo_der"]},
    "cadera_izq": {"puntos": ["hombro_izq", "cadera_izq", "rodilla_izq"]},
    "cadera_der": {"puntos": ["hombro_der", "cadera_der", "rodilla_der"]},
    "codo_izq": {"puntos": ["hombro_izq", "codo_izq", "muneca_izq"]},
    "codo_der": {"puntos": ["hombro_der", "codo_der", "muneca_der"]},
    "hombro_izq": {"puntos": ["cadera_izq", "hombro_izq", "codo_izq"]},
    "hombro_der": {"puntos": ["cadera_der", "hombro_der", "codo_der"]},
    "tobillo_izq": {"puntos": ["rodilla_izq", "tobillo_izq", "punta_pie_izq"]},
    "tobillo_der": {"puntos": ["rodilla_der", "tobillo_der", "punta_pie_der"]},
    "cabeza_tronco_izq": {"puntos": ["nariz", "hombro_izq", "cadera_izq"]},
    "cabeza_tronco_der": {"puntos": ["nariz", "hombro_der", "cadera_der"]},
}

MODEL_PATH = "models/pose_landmarker_full.task"
CAMERA_INDEX = 0
MIN_VISIBILIDAD = 0.1        # visibilidad mínima para DIBUJAR un punto (la de los ángulos está en utils.MIN_VISIBILIDAD)
INTERVALOS_PRINT = 10          # cada cuántos segundos se vuelcan coordenadas y ángulos por consola

INTERVALOS_COMPROBACION = 5

IDX = {nombre: i for i, nombre in enumerate(NOMBRES)}   # IDX["rodilla_der"] == 26

# BGR (por alguna razón)
COLOR_HUESO = (255, 0, 0)
COLOR_PUNTO_PREDETERMINADO = (0, 255, 255)
COLOR_PUNTO_CORRECTO = (0, 255, 0)
COLOR_PUNTO_INCORRECTO = (0, 0, 255)

UMBRAL_APROBADO = 80.0   # puntuación mínima (0-100) para dar la pose por buena
FACTOR_CAIDA = 3.0       # error (en múltiplos de la tolerancia) en el que la nota de un ángulo llega a 0 (> 1)


def _indice(p):
    """Acepta un índice (26) o un nombre ("rodilla_der") y devuelve el índice."""
    return IDX[p] if isinstance(p, str) else int(p)


def _no_relevante(datos):
    """True si el ángulo está marcado como no relevante en el .json ("objetivo": null)."""
    return "objetivo" in datos and datos["objetivo"] is None


def _angulos_relevantes(pose):
    """Pares (clave, datos) de los ángulos de la postura que no están marcados como no relevantes."""
    return [(clave, datos) for clave, datos in pose["angulos"].items() if not _no_relevante(datos)]

def _estado(observado, error, tolerancia):
    if not observado:
        return NO_EVALUABLE

    if abs(error) > tolerancia:
        return FALLO

    return OK

# ---------------------------------------------------------------------------
# Matemáticas
# ---------------------------------------------------------------------------
def teta(a, b, c):
    """
    Ángulo en grados (0-180) en el vértice b, formado por los segmentos b->a y b->c.
    Ejemplo rodilla: a = cadera, b = rodilla, c = tobillo. Funciona con puntos 2D o 3D.
    """
    a, b, c = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64), np.asarray(c, dtype=np.float64)
    v1, v2 = a - b, c - b          # el vértice es b (el punto central), no a
    coseno = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
    return float(np.degrees(np.arccos(np.clip(coseno, -1.0, 1.0))))


def matriz_de_putitos(landmarks):
    """Lista de landmarks (con .x .y .z) -> array (N, 3)."""
    return np.array([[p.x, p.y, p.z] for p in landmarks], dtype=np.float64)


def extraer_puntos(result):
    if not result.pose_landmarks:
        return None, None
    normalizados = result.pose_landmarks[0]
    visibilidad = np.array([p.visibility for p in normalizados], dtype=np.float64)

    if not result.pose_world_landmarks:
        return matriz_de_putitos(normalizados), visibilidad
    return matriz_de_putitos(result.pose_world_landmarks[0]), visibilidad


# ---------------------------------------------------------------------------
# Encuadre: ¿se ve el cuerpo que necesitamos?
# ---------------------------------------------------------------------------
def puntos_necesarios(pose):
    """Índices (ordenados, sin repetir) de los puntos que usan los ángulos RELEVANTES de una postura."""
    return sorted({_indice(p) for _, datos in _angulos_relevantes(pose) for p in datos["puntos"]})


def _lista_nombres(indices, maximo=4):
    nombres = [NOMBRES[i] for i in indices]
    return ", ".join(nombres[:maximo]) + (f" (+{len(nombres) - maximo})" if len(nombres) > maximo else "")


def comprobar_encuadre(result, indices):
    """
    Comprueba que todos los puntos de `indices` están DENTRO de la imagen y con visibilidad
    suficiente. Devuelve (ok, avisos). Los avisos van sin tildes para poder pintarlos con
    cv2.putText (que no dibuja bien los caracteres con tilde).

    Una coordenada normalizada fuera de [0, 1] significa que el punto está fuera de la imagen: el
    modelo lo está extrapolando, y esa posición no es de fiar aunque tenga un número asociado.
    """
    if not result.pose_landmarks:
        return False, ["No se detecta ninguna persona"]

    lm = result.pose_landmarks[0]
    m = MARGEN_ENCUADRE
    abajo, arriba, lados, poco = [], [], [], []
    for i in indices:
        p = lm[i]
        if p.y > 1 - m:
            abajo.append(i)
        elif p.y < m:
            arriba.append(i)
        elif p.x < m or p.x > 1 - m:
            lados.append(i)
        elif p.visibility < MIN_VISIBILIDAD:
            poco.append(i)

    avisos = []
    if abajo:
        avisos.append(f"Retrocede: no se ve abajo ({_lista_nombres(abajo)})")
    if arriba:
        avisos.append(f"Retrocede o baja la camara: no se ve arriba ({_lista_nombres(arriba)})")
    if lados:
        avisos.append(f"Centrate en la imagen ({_lista_nombres(lados)})")
    if poco:
        avisos.append(f"Poca visibilidad, mejora la luz o despeja ({_lista_nombres(poco)})")
    return (not avisos), avisos


# ---------------------------------------------------------------------------
# Medida y evaluación de ángulos
# ---------------------------------------------------------------------------
def medir_angulos(puntos, pose, visibilidad=None):
    """
    Mide los ángulos relevantes definidos en pose["angulos"] (se saltan los de "objetivo": null).
    Devuelve {clave: grados} o {clave: None} si algún punto del ángulo es poco visible.
    """
    medidos = {}
    for clave, datos in _angulos_relevantes(pose):
        ids = [_indice(p) for p in datos["puntos"]]
        if visibilidad is not None and min(visibilidad[i] for i in ids) < MIN_VISIBILIDAD:
            medidos[clave] = None
        else:
            medidos[clave] = teta(*puntos[ids])
    return medidos

def evaluar(medidos, pose):
    """
    Compara lo medido con lo esperado. Devuelve, por cada ángulo:
        {"medido", "objetivo", "tol", "error", "ok", "estado"}
    error = medido - objetivo (con signo: sirve para decir "abre más" / "cierra más").
    estado: OK, FALLO o NO_EVALUABLE. Si es NO_EVALUABLE, medido y error son None y ok es False.
    Los ángulos con "objetivo": null no aparecen en el resultado.
    """
    detalle = {}
    for clave, datos in _angulos_relevantes(pose):
        objetivo = datos["objetivo"]
        tol = datos.get("tol", datos.get("tolerancia"))   # el .json usa "tol"
        if tol is None:
            raise ValueError(f"El ángulo '{clave}' tiene objetivo pero no tolerancia ('tol') en el .json")
        medido = medidos.get(clave)
        if medido is None:
            detalle[clave] = {"medido": None, "objetivo": objetivo, "tol": tol,
                              "error": None, "ok": False, "estado": NO_EVALUABLE}
        else:
            error = medido - objetivo
            ok = abs(error) <= tol
            detalle[clave] = {"medido": medido, "objetivo": objetivo, "tol": tol,
                              "error": error, "ok": ok, "estado": OK if ok else FALLO}
    return detalle


def estado_postura(medidos, pose):
    """
    Estado global de la postura y detalle por ángulo. Devuelve (estado, detalle):
        NO_EVALUABLE si algún ángulo no se pudo medir (hay que pausar y pedir que se ajuste),
        FALLO        si todo se midió pero algún ángulo está fuera de tolerancia,
        OK           si todos los ángulos están dentro de tolerancia.
    NO_EVALUABLE tiene prioridad sobre FALLO: mientras falte información no se juzga al jugador.
    """
    detalle = evaluar(medidos, pose)
    if not detalle:
        raise ValueError("La postura no tiene ningún ángulo relevante (todos con 'objetivo': null)")
    estados = {d["estado"] for d in detalle.values()}
    if NO_EVALUABLE in estados:
        return NO_EVALUABLE, detalle
    if FALLO in estados:
        return FALLO, detalle
    return OK, detalle


def puntuar(medidos, pose):
    """
    Devuelve (aciertos, evaluables): ángulos dentro de tolerancia y ángulos que se pudieron medir.
    Los NO_EVALUABLE ya no cuentan como fallo: no entran en `evaluables`.
    """
    evaluables = [d for d in evaluar(medidos, pose).values() if d["estado"] != NO_EVALUABLE]
    return sum(d["ok"] for d in evaluables), len(evaluables)