from pathlib import Path
import json
import sys

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

import utils


def _cargar_conexiones():
    try:
        conexiones = mp.tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS
        return [(c.start, c.end) for c in conexiones]
    except AttributeError:
        return list(mp.solutions.pose.POSE_CONNECTIONS)


POSE_CONNECTIONS = _cargar_conexiones()


def cargar_postura_json(ruta_json):
    ruta = Path(ruta_json)
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró el archivo JSON: {ruta}")

    with ruta.open("r", encoding="utf-8") as f:
        data = json.load(f)

    postura = data.get("postura", data)
    if "angulos" not in postura:
        raise ValueError("El JSON no contiene la clave 'angulos' dentro de 'postura'.")
    return postura


def detectar_landmarks_imagen(ruta_imagen, ruta_modelo):
    imagen_bgr = cv2.imread(str(ruta_imagen))
    if imagen_bgr is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {ruta_imagen}")

    imagen_rgb = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imagen_rgb)

    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(ruta_modelo)),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
    )

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        resultado = landmarker.detect(mp_image)

    if not resultado.pose_landmarks:
        raise ValueError("MediaPipe no detectó ninguna persona en la imagen.")

    return imagen_bgr, resultado


def puntos_en_json(postura):
    indices = set()
    for datos in postura.get("angulos", {}).values():
        for p in datos.get("puntos", []):
            indices.add(utils._indice(p))
    return indices


def crear_panel_esqueleto(alto, ancho, resultado, postura, min_vis=0.1):
    panel = np.full((alto, ancho, 3), (24, 24, 24), dtype=np.uint8)
    landmarks = resultado.pose_landmarks[0]

    coords_2d = [(int(p.x * ancho), int(p.y * alto)) for p in landmarks]
    visibles = [p.visibility >= min_vis for p in landmarks]
    indices_json = puntos_en_json(postura)

    for i, j in POSE_CONNECTIONS:
        if visibles[i] and visibles[j]:
            cv2.line(panel, coords_2d[i], coords_2d[j], (255, 120, 0), 2)

    for idx, (x, y) in enumerate(coords_2d):
        if not visibles[idx]:
            continue
        en_json = idx in indices_json
        color = (0, 255, 255) if en_json else (0, 255, 0)
        radio = 6 if en_json else 4
        cv2.circle(panel, (x, y), radio, color, -1)

    return panel


def redimensionar_para_pantalla(imagen, max_ancho=1600, max_alto=900):
    alto, ancho = imagen.shape[:2]
    if ancho <= max_ancho and alto <= max_alto:
        return imagen

    escala = min(max_ancho / ancho, max_alto / alto)
    nuevo_ancho = max(1, int(ancho * escala))
    nuevo_alto = max(1, int(alto * escala))
    return cv2.resize(imagen, (nuevo_ancho, nuevo_alto), interpolation=cv2.INTER_AREA)


def mostrar_json_lado_a_lado(
    ruta_json,
    ruta_imagen,
    ruta_modelo=None,
    ruta_salida_img=None,
    max_ancho=1600,
    max_alto=900,
):
    if ruta_modelo is None:
        ruta_modelo = "models/pose_landmarker_full.task"

    postura = cargar_postura_json(ruta_json)
    imagen_bgr, resultado = detectar_landmarks_imagen(ruta_imagen, ruta_modelo)
    alto, ancho = imagen_bgr.shape[:2]

    panel_esqueleto = crear_panel_esqueleto(alto, ancho, resultado, postura)
    composicion = np.hstack((imagen_bgr, panel_esqueleto))

    if ruta_salida_img:
        salida = Path(ruta_salida_img)
        salida.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(salida), composicion)

    vista = redimensionar_para_pantalla(composicion, max_ancho=max_ancho, max_alto=max_alto)
    titulo = f"Comparativa JSON - {postura.get('nombre', 'postura')}"
    cv2.namedWindow(titulo, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(titulo, vista)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return composicion


if __name__ == "__main__":
    mostrar_json_lado_a_lado(
        ruta_json="jason/facil/debug.json",
        ruta_imagen="media/facil/debug.jpg",
    )