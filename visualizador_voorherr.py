from pathlib import Path
import json
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from utils import indice, MODEL_PATH

CONECTORES = [(c.start, c.end) for c in mp.tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS]

def cargar_postura(ruta):
    """
    Carga las posturas de la ruta recibida. Se asume que la ruta es correcta y sin errores porque lo usamos
    para trazabilidad y debug
    """
    ruta = Path(ruta)

    with ruta.open("r", encoding="utf-8") as f:
        datos = json.load(f)

    return datos["postura"]

def detectar_exoesqueleto_imagen(ruta_imagen, ruta_modelo):
    """
    Obtiene la ruta de la imagen y del modelo a usar. Toma la imagen, la carga (pasando a RGB) y se lo pasamos al modelo
    como si fuera un frame de la cámara para que detecte los puntos del cuerpo. 
    """
    imagen_bgr = cv2.imread(ruta_imagen)

    imagen_rgb = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB) # No entiendo porqué pero cv2.imread() lee la paleta en el orden BGR. Esto lo pasa al clásico RGB
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imagen_rgb)

    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(ruta_modelo)),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
    )

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        resultado = landmarker.detect(mp_image)

    return imagen_bgr, resultado

def detectar_puntos_json(postura):
    """
    Se obtienen todos los puntos usados en la postura JSON. Toma los nombres de los vertices y obitiene el índice
    correspondiente del esqueleto. Por eso la funcion es un poco rara.
    """
    indices = set()
    angulos = postura["angulos"] if postura["angulos"] else {}
    
    for datos in angulos.values():
        puntos = datos["puntos"] if datos["puntos"] else []
        for p in puntos:
            indices.add(indice(p))
    
    return indices

def crear_panel_esqueleto(altura, anchura, resultado, postura, minima_visibiliad=0.1):
    """
    Crea la imagen del esqueleto de la foto usada donde meustra los puntos usados en el JSON y lo que no. Sirve para 
    debuguear si la imagen ha sido capturada como se esperaba.
    """
    panel = np.full((altura, anchura, 3), (24, 24, 24), dtype=np.uint8)
    landmarks = resultado.pose_landmarks[0]

    coordenadas_2d = [(int(punto.x * anchura), int(punto.y * altura)) for punto in landmarks]
    visibles = [punto.visibility >= minima_visibiliad for punto in landmarks]

    for i, j in CONECTORES:
        if visibles[i] and visibles[j]:
            cv2.line(panel, coordenadas_2d[i], coordenadas_2d[j], (255, 120, 0), 2) #Color azul clarito, grosor de 2

    indices_jason = detectar_puntos_json(postura)
    for indice, (x, y) in enumerate(coordenadas_2d):
        if not visibles[indice]:
            continue

        if indice in indices_jason:
            color, radio = (0, 255, 255), 6 #Color (amarillo) y radio. Muesta los puntos detectados en el JSON
        else:
            color, radio = (0, 255, 0), 4 # Color (azul) y radio menor. Muestra puntos no detectados en el JSON

        cv2.circle(panel, (x, y), radio, color, -1) # Grosor -1 porque si le das algún valor positivo solo dibuja la circunferencia

    return panel

def redimensionar_pantalla(imagen, max_ancho=1600, max_alto=900):
    """
    Redimensiona la pantalla si la imagen se sobrepasa de los límites.
    """
    alto, ancho = imagen.shape[0], imagen.shape[1]
    if ancho <= max_ancho and alto <= max_alto:
        return imagen

    escala = min(max_alto / ancho, max_alto / alto)
    ancho_nuevo, alto_nuevo = max(1, int(ancho * escala)), max(1, int(alto * escala))
    return cv2.resize(imagen, (ancho_nuevo, alto_nuevo), interpolation=cv2.INTER_AREA)

def mostrar_json_lado_a_lado(ruta_json, ruta_imagen, max_ancho=1600, max_alto=900):
    """
    Muestra dos imágenes. A la derecha la imagen que tomamos y a la izquierda su esqueleto detectado por el modelo que
    empleamos. Puede no ser 100% exacto por culpa de la imprecisión del modelo. Por eso, creamos esta función para debuguear si
    los JSON capturaron bien la postura.
    """

    postura = cargar_postura(ruta_json)
    imagen_bgr, resultado = detectar_exoesqueleto_imagen(ruta_imagen, MODEL_PATH)
    alto, ancho = imagen_bgr.shape[0], imagen_bgr.shape[0]

    panel_esqueleto = crear_panel_esqueleto(alto, ancho, resultado, postura)
    composicion = np.hstack((imagen_bgr, panel_esqueleto))

    vista = redimensionar_pantalla(composicion, max_ancho=max_ancho, max_alto=max_alto)
    titulo = f"Comparativa JSON - {postura["nombre"] if postura["nombre"] else "postura"}"
    cv2.namedWindow(titulo, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(titulo, vista)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return composicion

if __name__ == "__main__":
    dificultad = "dificil"
    imagen = "p4"
    mostrar_json_lado_a_lado(
        ruta_json=f"jason/{dificultad}/{imagen}.json",
        ruta_imagen=f"media/{dificultad}/{imagen}.jpg",
    )