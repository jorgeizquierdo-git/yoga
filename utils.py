import numpy as np

# Configuración global para todos los archivos

# Un punto se considera "fuera de la imagen" si queda a menos de este margen del borde (0-1).
MARGEN_ENCUADRE = 0.02 # El margen de separación fuera de la imagen. Si se va, no se muestra

# Nombres de los 33 landmarks de MediaPipe Pose. El índice de la lista corresponde al índice del landmark
NOMBRES = [
    "nariz", "ojo_izq_int", "ojo_izq", "ojo_izq_ext", "ojo_der_int", "ojo_der", "ojo_der_ext", "oreja_izq", "oreja_der",
    "boca_izq", "boca_der", "hombro_izq", "hombro_der", "codo_izq", "codo_der", "muneca_izq", "muneca_der", "meñique_izq", "meñique_der",
    "indice_izq", "indice_der", "pulgar_izq", "pulgar_der", "cadera_izq", "cadera_der", "rodilla_izq", "rodilla_der", 
    "tobillo_izq", "tobillo_der", "talon_izq", "talon_der", "punta_pie_izq", "punta_pie_der",
]
IDX = {nombre: i for i, nombre in enumerate(NOMBRES)}

# Angulos de cada punto y los puntos que se usan para su cálculo. Estructura necesaria para coincidir con nuestros JSON
ANGULOS = {
    "angulos": {
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
}

MODEL_PATH = "models/pose_landmarker_full.task"
CAMERA_INDEX = 0 # 1 si no te funciona el predeterminado

MIN_VISIBILIDAD = 0.3 # minima visibilidad para considrerar le punto

INTERVALOS_COMPROBACION = 5 # el intervalo de tiempo para comprobar la postura con la seleccionada durante el debug

NOMBRES_EN_PANTALLA = {"rodilla", "cadera", "tobillo", "hombro", "codo"} # Nombre de los vertices que nos interesaba analizar durante el desarrollo
COLOR_TEXTO = (255, 255, 255) # Color para estos mensajes de debug

# BGR (por alguna razón)
COLOR_HUESO = (255, 0, 0)
COLOR_PUNTO_PREDETERMINADO = (255, 200, 120) # azul claro para los puntos que no se evaluan
COLOR_PUNTO_CORRECTO = (0, 255, 0) # verde para puntos con buen ángulo
COLOR_PUNTO_CERCA = (0, 255, 255) # amarillo para los puntos cerca de tener buen angulo
COLOR_PUNTO_INCORRECTO = (0, 0, 255) # rojo para los puntos con mal ángulo.

MARGEN_CERCA = 15.0 # Angulos superiores a la tolerancia de dicho punto en la cual se muestra amarillo


FILTRO_ACTIVO = True # False para ver el esqueleto "crudo" de MediaPipe y comparar
FILTRO_MIN_CUTOFF = 1.0 # Hz. Más bajo = suaviza más estando quieto (pero tarda más en llegar a la pose)
FILTRO_BETA = 5.0 # Más alto = menos retraso cuando te mueves rápido (pero tiembla más al moverse)
FILTRO_D_CUTOFF = 1.0 # Hz. Suavizado de la velocidad que usa el filtro; normalmente no se toca
SALTO_MAXIMO = 0.5 # desplazamiento máximo de un punto en UN fotograma, en "troncos" (hombro-cadera)
MANTENER_MAX_S = 0.5 # durante cuánto se ignora un salto o una caída de visibilidad antes de aceptarla

UMBRAL_APROBADO = 70.0   # puntuación global mínima (0-100) para dar la pose por buena

T_EXPLICACION = 3  # Segundos para mostrar la imagen
T_LIMITE = {"facil": 30, "medio": 45, "dificil": 60} # Distintos tiempos para hacer las posturas para cada dificultad
VENTANA_PUNTUACION = 5 # segundos del mejor tramo del usuario durante ese periodo de tiempo
GRACIA_MANTENER = 0.5 # segundos por debajo del umbral que se perdonan al contar cuánto lleva mantenida la pose
COBERTURA_VENTANA = 0.5 # fracción mínima de fotogramas evaluables en un tramo para que ese tramo cuente
T_RESULTADO_POSE = 2 # segundos para mostrar el resultado de la posicion

SONIDOS_ACTIVOS = True     # False para jugar en silencio
CARPETA_SONIDOS = "soniditos"
AVISO_AUSENCIA_CADA = 2.5  # intervalo de tiempo para volver a ejecutar el sonido de que el usuario está fuera de la cámara durante el juego
AUSENCIA_MINIMA = 0.5 # segundos sin detectar al jugador antes del primer zumbido (evita avisos por parpadeos)
TICS_FINALES = 5 # numero de tics en cada uno de los últimos segundos del tiempo límite

# Transcripción nombres de los archivos a sus nombres reales (para mostrarlos durante el juego)
NOMBRES_POSES = {
    "facil/default": "T", 
    "facil/p1": "Arbol", 
    "facil/p2": "Guerrero II", 
    "facil/p3": "Silla",
    "facil/p4": "Perro boca abajo",
    "medio/default": "Balasana", 
    "medio/p1": "Puente", 
    "medio/p2": "Cobra", 
    "medio/p3": "Triangulo",
    "medio/p4": "Lagarto",
    "dificil/default": "Media luna", 
    "dificil/p1": "Zancada con estiramiento", 
    "dificil/p2": "Invertida plegada",
    "dificil/p3": "Invertida con pierna arriba", 
    "dificil/p4": "Wild thing"
}

PESO_POR_DEFECTO = 1.0 # Ponderación por defecto de cada ángulo. Varía según su importancia

# Evaluación parcial: si algún ángulo no se puede medir (punto fuera de la imagen o poco visible) se puntúa con
# los demás, siempre que lo medido sume al menos esta fracción del peso total de la postura. Por debajo, la pose
# es NO_EVALUABLE (no se juzga al jugador: hay que pedirle que se recoloque).
COBERTURA_MINIMA = 0.6 # fracción del peso total de la postura que deben superar el resto de ángulos para evaluar la postura. Se hace para 
# evitar que algún punto que no se pueda evaluar por cula del modelo penalice al jugador. Preferimos ayudarle
ERROR_MAXIMO = 60.0 # error (en grados) a partir del cual un ángulo puntúa 0; con error 0 puntúa 1 y baja linealmente


def indice(p):
    """
    Recibe ya sea el índice o el nombre del punto y devuelve su índice. Se tomó esta decisión para evitar problemas con nuestro
    caos a la hora de programar.
    """
    return IDX[p] if isinstance(p, str) else int(p)

def no_relevante(datos):
    """
    Devuelve True si el ángulo está marcado como 'objetivo:null
    """
    return "objetivo" in datos and datos["objetivo"] is None


def angulos_relevantes(pose):
    """
    Sevuelve el par (clave, datos) de los angulos de la postura que no están marcados como irrelevantes
    """
    return [(clave, datos) for clave, datos in pose["angulos"].items() if not no_relevante(datos)]

def teta(a, b, c):
    """
    Ángulo en grados sobre el vertice b apoyandose sobre los otros dos vertices a, c.
    """
    a, b, c = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64), np.asarray(c, dtype=np.float64)
    v1, v2 = a - b, c - b 
    coseno = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
    return float(np.degrees(np.arccos(np.clip(coseno, -1.0, 1.0))))


def matriz_de_putitos(landmarks, ancho, alto):
    """
    Lista de puntos. Genera una matriz (N, 2)
    """
    return np.array([[puntos.x * ancho, puntos.y * alto] for puntos in landmarks], dtype=np.float64)


def extraer_puntos(result, ancho, alto):
    """
    Devuelve la matriz de puntos y su visibilidad.
    """
    if not result.pose_landmarks:
        return None, None
    
    normalizados = result.pose_landmarks[0]
    visibilidad = np.array([p.visibility for p in normalizados], dtype=np.float64)
    
    return matriz_de_putitos(normalizados, ancho, alto), visibilidad


def esqueleto_a_dict(result, ancho, alto):
    """
    Guarda el esqueleto en un diccionario. Sirve para el guardado de posturas en JSON
    """
    puntos, visibilidad = extraer_puntos(result, ancho, alto)

    if puntos is None:
        return None
    
    return {
        "ancho": int(ancho),
        "alto": int(alto),
        "puntos": [
            {
                "idx": i, 
                "nombre": NOMBRES[i], 
                "x": round(float(x), 1), 
                "y": round(float(y), 1), 
                "visibilidad": round(float(v), 3)
            } for i, ((x, y), v) in enumerate(zip(puntos, visibilidad))
        ],
    }

def peso_de(clave, pesos=None):
    """
    Devuelve el peso de la clave del diccionario. Por defecto, devuelve el parametro establecido PESO_POR_DEFECTO.
    Primero pruba con la clave entera. Si es de tipo "rodilla_derecha" busca el punto "rodilla"
    """
    pesos = pesos if pesos is not None else {}
    return float(pesos.get(clave, pesos.get(clave.rsplit("_", 1)[0], PESO_POR_DEFECTO)))
    # clave.rsplit("_", 1)[0] para obtener los nombres de los puntos de la forma "cadera_izquierda" -> "cadera"


def puntos_fuera_de_imagen(result):
    """
    Indices que quedan fuera del encuadre
    """
    if not result.pose_landmarks:
        return set()

    return {i for i, punto in enumerate(result.pose_landmarks[0]) 
            if not (MARGEN_ENCUADRE <= punto.x <= 1 - MARGEN_ENCUADRE and MARGEN_ENCUADRE <= punto.y <= 1 - MARGEN_ENCUADRE)}


def puntos_necesarios(pose):
    """Índices (ordenados, sin repetir) de los puntos que usan los ángulos RELEVANTES de una postura."""
    return sorted({indice(p) for _, datos in angulos_relevantes(pose) for p in datos["puntos"]})


def lista_nombres(indices, maximo=4):
    """
    Devuelve una lista de los nombres de los índices del esqueleto
    """
    nombres = [NOMBRES[i] for i in indices]
    return ", ".join(nombres[:maximo]) + (f" (+{len(nombres) - maximo})" if len(nombres) > maximo else "")


def comprobar_encuadre(result, indices):
    """
    Comprueba que todos los puntos están dentro de la imagen y con visibilidad. Devuelve (ok, avisos). Los avisos
    son para mostrarlos en pantalla
    """

    if not result.pose_landmarks:
        return False, ["No se detecta ninguna persona"]

    landmarks = result.pose_landmarks[0]

    abajo, arriba, lados, poco = [], [], [], []
    for i in indices:
        punto = landmarks[i]
        if punto.y > 1 - MARGEN_ENCUADRE:
            abajo.append(i)
        elif punto.y < MARGEN_ENCUADRE:
            arriba.append(i)
        elif punto.x < MARGEN_ENCUADRE or punto.x > 1 - MARGEN_ENCUADRE:
            lados.append(i)
        elif punto.visibility < MIN_VISIBILIDAD:
            poco.append(i)

    avisos = []
    if abajo:
        avisos.append(f"Retrocede: no se ve abajo ({lista_nombres(abajo)})")
    if arriba:
        avisos.append(f"Retrocede o baja la camara: no se ve arriba ({lista_nombres(arriba)})")
    if lados:
        avisos.append(f"Centrate en la imagen ({lista_nombres(lados)})")
    if poco:
        avisos.append(f"Poca visibilidad, mejora la luz o despeja ({lista_nombres(poco)})")
    
    return (not avisos), avisos # Sin avisos -> True

def medir_angulos(puntos, pose, visibilidad=None):
    """
    Mide los ángulos relevantes (no cuenta los 'objetivo':null). Devuelve los grados del punto o None si no tenía suficiente visibilidad
    """

    medidos = {}
    for clave, datos in angulos_relevantes(pose):
        ids = [indice(p) for p in datos["puntos"]]

        if visibilidad is not None and min(visibilidad[i] for i in ids) < MIN_VISIBILIDAD: # Si uno de los puntos no es lo suficientemente visible
            medidos[clave] = None
        
        else:
            medidos[clave] = teta(*puntos[ids])
    
    return medidos

def evaluar(medidos, pose):
    """
    Compara la posicion medida con la esperada. Los angulos sin objetivo no aparecen en el resultado
    """
    
    detalle = {}
    for clave, datos in angulos_relevantes(pose):
        objetivo = datos["objetivo"]
        tol = datos["tol"]
        medido = medidos.get(clave, None)

        if medido is None:
            detalle[clave] = {
                "medido": None,
                "objetivo": objetivo, 
                "tol": tol,
                "error": None, 
                "ok": False, 
                "estado": "no_evaluable"
            }
        
        else:
            error = medido - objetivo
            ok = abs(error) <= tol
            detalle[clave] = {
                "medido": medido, 
                "objetivo": objetivo, 
                "tol": tol,
                "error": error, 
                "ok": ok, 
                "estado": "ok" if ok else "fallo"
            }
    
    return detalle


def estado_postura(medidos, pose):
    """
    Estado global de la postura y de sus ángulos. Devuelve una tupla (estado, detalle)
    """
    detalle = evaluar(medidos, pose)

    if not detalle:
        raise ValueError("La postura no tiene ningún ángulo relevante (todos con 'objetivo': null)")

    estados = {dato["estado"] for dato in detalle.values()}

    # Si recibe cualquier "no evaluable" o "fallo" se considera de manera global
    if "no_evaluable" in estados:
        return "no_evaluable", detalle
    if "fallo" in estados:
        return "fallo", detalle
    return "ok", detalle


def puntuar(medidos, pose):
    """
    Devuelve los aciertos y su evaluación de cada punto sin estado 'no_evaluable'
    """
    evaluables = [dato for dato in evaluar(medidos, pose).values() if dato["estado"] != "no_evaluable"]
    return sum(dato["ok"] for dato in evaluables), len(evaluables)