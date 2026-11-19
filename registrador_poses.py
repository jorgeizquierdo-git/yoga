import json
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
<<<<<<< HEAD:Carpeta_Jorge/registrador_poses.py
from pathlib import Path
import sys
=======
import numpy as np
from utils import teta
>>>>>>> 1b4563b (los he sacado fuera de las carpetas. Así no hay problemas de directorios):registrador_poses.py

RAIZ_YOGA = Path(__file__).resolve().parent.parent
CARPETA_JULIO = RAIZ_YOGA / "Carpeta_julio"

for ruta in (RAIZ_YOGA, CARPETA_JULIO):
    ruta_str = str(ruta)
    if ruta_str not in sys.path:
        sys.path.insert(0, ruta_str)

import utils
from utils import extraer_puntos, medir_angulos

ANGULOS_ESTANDAR = {
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


def detectar_pose(ruta_imagen, ruta_modelo):
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

    return resultado


def registrar_pose_desde_imagen(
    ruta_imagen,
    ruta_modelo,
    ruta_salida_json,
    nombre_pose,
    articulaciones_relevantes,
    tolerancia=10,
    mantener_s=10,
):
    resultado = detectar_pose(ruta_imagen=ruta_imagen, ruta_modelo=ruta_modelo)
    puntos, visibilidad = extraer_puntos(resultado)
    if puntos is None:
        raise ValueError("No se pudieron extraer puntos válidos de la detección.")

    relevantes_set = set(articulaciones_relevantes)
    plantilla = {"angulos": ANGULOS_ESTANDAR}
    angulos_calculados = medir_angulos(puntos, plantilla, visibilidad)

    angulos_json = {}
    for clave, datos in ANGULOS_ESTANDAR.items():
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
        }
    }

    salida = Path(ruta_salida_json)
    salida.parent.mkdir(parents=True, exist_ok=True)
    with salida.open("w", encoding="utf-8") as f:
        json.dump(contenido, f, indent=2, ensure_ascii=False)

    return contenido


if __name__ == "__main__":
    registrar_pose_desde_imagen( #deberías cabiar los valores de las rutas si vas a cambiar de representacón
        ruta_imagen="Carpeta_Jorge/media/p1.jpg",
        ruta_modelo="models/pose_landmarker_full.task",
<<<<<<< HEAD:Carpeta_Jorge/registrador_poses.py
        ruta_salida_json="Carpeta_Jorge/jason/ejemplo1.json",
        nombre_pose="ejemplo",
        articulaciones_relevantes=list(ANGULOS_ESTANDAR.keys()),
        tolerancia=10,
    )
=======
        ruta_salida_json="",
        nombre_pose="",
        articulaciones_relevantes=[],
        tolerancia=0
    )
    
    
    
    
    
>>>>>>> 1b4563b (los he sacado fuera de las carpetas. Así no hay problemas de directorios):registrador_poses.py
