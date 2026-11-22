"""
Archivo para funciones generales y de ayuda: configuración global y funciones matemáticas.

Flujo previsto en el juego:
    result = landmarker.detect_for_video(...)

    # 1) Antes de empezar (y durante el juego): ¿se ve el cuerpo entero?
    ok, avisos = comprobar_encuadre(result, puntos_necesarios(pose))

    # 2) Medir y evaluar la postura
    puntos, visib = extraer_puntos(result, ancho, alto) # (33, 2) en píxeles y (33,)
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
# Los ángulos se calculan en 2D, con las coordenadas (x, y) en PÍXELES de la imagen: x·ancho, y·alto.
# Se descarta la z (profundidad), que es la coordenada que peor estima MediaPipe, y no se usan las
# normalizadas (0-1) tal cual porque x e y irían en unidades distintas y deformarían los ángulos.
# Así el ángulo calculado es exactamente el que se ve dibujado en pantalla.

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

ANGULOS = {"angulos": {
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
}}

MODEL_PATH = "models/pose_landmarker_full.task"
CAMERA_INDEX = 0
MIN_VISIBILIDAD = 0.3        # visibilidad mínima para DIBUJAR un punto (la de los ángulos está en utils.MIN_VISIBILIDAD)
INTERVALOS_PRINT = 10          # cada cuántos segundos se vuelcan coordenadas y ángulos por consola

INTERVALOS_COMPROBACION = 5
INTERVALO_CAPTURA = 20         # cada cuántos segundos se captura el esqueleto y se guarda en un .json
CARPETA_CAPTURAS = "jason/capturas"

# Articulaciones cuyo NOMBRE se escribe en pantalla (las que más guerra dan). El valor del ángulo se
# escribe en todos los vértices de ANGULOS; el nombre, solo en estos.
NOMBRES_EN_PANTALLA = {"rodilla", "cadera", "tobillo", "hombro", "codo"}
COLOR_TEXTO = (255, 255, 255)

IDX = {nombre: i for i, nombre in enumerate(NOMBRES)}   # IDX["rodilla_der"] == 26

# BGR (por alguna razón)
COLOR_HUESO = (255, 0, 0)
COLOR_PUNTO_PREDETERMINADO = (255, 200, 120)   # azul claro: puntos que la pose no evalúa
COLOR_PUNTO_CORRECTO = (0, 255, 0)             # verde:    |error| <= tol
COLOR_PUNTO_CERCA = (0, 255, 255)              # amarillo: tol < |error| <= tol + MARGEN_CERCA
COLOR_PUNTO_INCORRECTO = (0, 0, 255)           # rojo:     |error| > tol + MARGEN_CERCA, o no se pudo medir

MARGEN_CERCA = 15.0      # grados por encima de la tolerancia en los que el ángulo se pinta en amarillo ("casi")
# --- Reducción de ruido del esqueleto en tiempo real (suavizado.py) ---
FILTRO_ACTIVO = True       # False para ver el esqueleto "crudo" de MediaPipe y comparar
FILTRO_MIN_CUTOFF = 1.0    # Hz. Más bajo = suaviza más estando quieto (pero tarda más en llegar a la pose)
FILTRO_BETA = 5.0          # Más alto = menos retraso cuando te mueves rápido (pero tiembla más al moverse)
FILTRO_D_CUTOFF = 1.0      # Hz. Suavizado de la velocidad que usa el filtro; normalmente no se toca
SALTO_MAXIMO = 0.5         # desplazamiento máximo de un punto en UN fotograma, en "troncos" (hombro-cadera)
MANTENER_MAX_S = 0.5       # durante cuánto se ignora un salto o una caída de visibilidad antes de aceptarla

ESCALA_VENTANA = 2       # la ventana del tracker se abre a este múltiplo del tamaño de la cámara (y se puede estirar)

UMBRAL_APROBADO = 70.0   # puntuación global mínima (0-100) para dar la pose por buena

# --- Juego (game.py / ui.py / main.py) ---
T_EXPLICACION = 3          # s mostrando la foto de la pose a pantalla grande
# Tiempo máximo para hacer cada postura, según la dificultad: las difíciles necesitan más para colocarse
# (equilibrio, piernas...). Se puntúa desde el principio y la postura termina antes en cuanto se mantiene
# >= UMBRAL_APROBADO durante VENTANA_PUNTUACION s seguidos.
T_LIMITE = {"facil": 30, "medio": 45, "dificil": 60}
VENTANA_PUNTUACION = 5     # nota de la pose = media del MEJOR tramo de estos segundos seguidos
GRACIA_MANTENER = 0.5      # s por debajo del umbral que se perdonan al contar cuánto lleva mantenida la pose
COBERTURA_VENTANA = 0.5    # fracción mínima de fotogramas evaluables en un tramo para que ese tramo cuente
T_RESULTADO_POSE = 2       # s mostrando la nota de una pose antes de pasar a la siguiente

# Nombres que se muestran en pantalla (sin tildes: OpenCV no las dibuja). Si falta uno, se usa "Postura N".
NOMBRES_POSES = {
    "facil/default": "T", "facil/p1": "Arbol", "facil/p2": "Guerrero II", "facil/p3": "Silla",
    "facil/p4": "Perro boca abajo",
    "medio/default": "Balasana", "medio/p1": "Puente", "medio/p2": "Cobra", "medio/p3": "Triangulo",
    "medio/p4": "Lagarto",
    "dificil/default": "Media luna", "dificil/p1": "Zancada con estiramiento", "dificil/p2": "Invertida plegada",
    "dificil/p3": "Invertida con pierna arriba", "dificil/p4": "Wild thing",
}

# Ponderaciones: cada ángulo de una postura puede llevar "peso" en su .json (si no lo lleva, vale PESO_POR_DEFECTO).
# La puntuación es la media PONDERADA de las notas de los ángulos medidos.
PESO_POR_DEFECTO = 1.0

# Evaluación parcial: si algún ángulo no se puede medir (punto fuera de la imagen o poco visible) se puntúa con
# los demás, siempre que lo medido sume al menos esta fracción del peso total de la postura. Por debajo, la pose
# es NO_EVALUABLE (no se juzga al jugador: hay que pedirle que se recoloque).
COBERTURA_MINIMA = 0.6
ERROR_MAXIMO = 60.0      # error (en grados) a partir del cual un ángulo puntúa 0; con error 0 puntúa 1 y baja linealmente


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
    if observado is None:
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


def matriz_de_putitos(landmarks, ancho, alto):
    """Lista de landmarks normalizados (con .x .y) -> array (N, 2) en píxeles: (x·ancho, y·alto)."""
    return np.array([[p.x * ancho, p.y * alto] for p in landmarks], dtype=np.float64)


def extraer_puntos(result, ancho, alto):
    """
    Devuelve (puntos, visibilidad): puntos es (33, 2) en píxeles y visibilidad (33,).
    ancho y alto son los de la imagen que se pasó a MediaPipe (frame.shape[1], frame.shape[0]).
    """
    if not result.pose_landmarks:
        return None, None
    normalizados = result.pose_landmarks[0]
    visibilidad = np.array([p.visibility for p in normalizados], dtype=np.float64)
    return matriz_de_putitos(normalizados, ancho, alto), visibilidad


def esqueleto_a_dict(result, ancho, alto):
    """
    Los 33 puntos del esqueleto en píxeles, para guardarlos en un .json y poder reconstruir la postura:
        {"ancho": w, "alto": h, "puntos": [{"idx", "nombre", "x", "y", "visibilidad"}, ...]}
    """
    puntos, visibilidad = extraer_puntos(result, ancho, alto)
    if puntos is None:
        return None
    return {
        "ancho": int(ancho),
        "alto": int(alto),
        "puntos": [{"idx": i, "nombre": NOMBRES[i], "x": round(float(x), 1), "y": round(float(y), 1),
                    "visibilidad": round(float(v), 3)}
                   for i, ((x, y), v) in enumerate(zip(puntos, visibilidad))],
    }


# ---------------------------------------------------------------------------
# Encuadre: ¿se ve el cuerpo que necesitamos?
# ---------------------------------------------------------------------------
def peso_de(clave, pesos=None):
    """
    Peso de un ángulo según un diccionario de pesos que admite la clave exacta ("rodilla_der") o el tipo
    ("rodilla", vale para los dos lados). La clave exacta manda sobre el tipo.
    """
    pesos = pesos or {}
    return float(pesos.get(clave, pesos.get(clave.rsplit("_", 1)[0], PESO_POR_DEFECTO)))


def puntos_fuera_de_imagen(result):
    """Índices de los puntos que caen fuera de la imagen (con el margen MARGEN_ENCUADRE): no son de fiar."""
    if not result.pose_landmarks:
        return set()
    m = MARGEN_ENCUADRE
    return {i for i, p in enumerate(result.pose_landmarks[0])
            if not (m <= p.x <= 1 - m and m <= p.y <= 1 - m)}


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