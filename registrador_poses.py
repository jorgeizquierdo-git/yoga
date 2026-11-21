import json
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

from utils import extraer_puntos, medir_angulos, esqueleto_a_dict, ANGULOS


def detectar_pose(ruta_imagen, ruta_modelo):
    """
    ruta_imagen puede ser:
      - una ruta a un archivo de imagen (str o Path), como hasta ahora, o
      - una imagen ya cargada en memoria (np.ndarray BGR, tal cual la da OpenCV / cap.read()).
    Así la captura de la cámara reutiliza exactamente el mismo camino que las fotos, sin guardar nada en disco.

    Devuelve (resultado, ancho, alto): el tamaño hace falta para pasar los puntos a píxeles.
    """
    if isinstance(ruta_imagen, np.ndarray):
        imagen_bgr = ruta_imagen
    else:
        imagen_bgr = cv2.imread(str(ruta_imagen))
        if imagen_bgr is None:
            raise FileNotFoundError(f"No se ha podido encontrar la imagen: {ruta_imagen}")

    imagen_rgb = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imagen_rgb)

    base_options = python.BaseOptions(model_asset_path=str(ruta_modelo))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
    )

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        resultado = landmarker.detect(mp_image)

    if not resultado.pose_landmarks:
        raise ValueError("El modelo no ha detectado ninguna persona en la imagen.")

    alto, ancho = imagen_bgr.shape[:2]
    return resultado, ancho, alto


def registrar_pose_desde_imagen(
    ruta_imagen,
    ruta_modelo,
    ruta_salida_json,
    nombre_pose,
    articulaciones_relevantes,
    tolerancia=10,
    mantener_s=10,
):
    resultado, ancho, alto = detectar_pose(ruta_imagen=ruta_imagen, ruta_modelo=ruta_modelo)
    puntos, visibilidad = extraer_puntos(resultado, ancho, alto)
    if puntos is None:
        raise ValueError("No se pudieron extraer puntos válidos de la detección.")

    relevantes_set = set(articulaciones_relevantes)
    plantilla = ANGULOS                       # ya tiene la forma {"angulos": {...}}
    angulos_calculados = medir_angulos(puntos, plantilla, visibilidad)

    angulos_json = {}
    for clave, datos in ANGULOS["angulos"].items():
        es_relevante = clave in relevantes_set
        objetivo = round(float(angulos_calculados[clave]), 1) if (es_relevante and angulos_calculados.get(clave) is not None) else None
        angulos_json[clave] = {
            "puntos": list(datos["puntos"]),
            "objetivo": objetivo,
            "tol": tolerancia if es_relevante else None,
        }

    contenido = {
        "postura": {
            "nombre": nombre_pose,
            "mantener_s": mantener_s,
            "angulos": angulos_json,
        },
        # Los 33 puntos en píxeles, para poder reconstruir el esqueleto. El juego solo lee "postura".
        "esqueleto": esqueleto_a_dict(resultado, ancho, alto),
    }

    salida = Path(ruta_salida_json)
    salida.parent.mkdir(parents=True, exist_ok=True)
    with salida.open("w", encoding="utf-8") as f:
        json.dump(contenido, f, indent=2, ensure_ascii=False)

    return contenido


if __name__ == "__main__":
    registrar_pose_desde_imagen( #deberías cabiar los valores de las rutas si vas a cambiar de representacón
        ruta_imagen="media/dificil/p11.jpg",
        ruta_modelo="models/pose_landmarker_full.task",
        ruta_salida_json="jason/dificil/p11.json",
        nombre_pose="ejemplo",
        articulaciones_relevantes=list(ANGULOS["angulos"].keys()),
        tolerancia=10,
    )