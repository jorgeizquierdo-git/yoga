import json
from pathlib import Path
import cv2
import mediapipe as mp
from utils import ANGULOS, MODEL_PATH, extraer_puntos, medir_angulos, esqueleto_a_dict, peso_de


def detectar_pose(ruta_imagen, ruta_modelo):
    """
    Detecta el esqueleto de una foto. Es nuestra forma de inferir en las imagenes para
    obtener los puntos que el usuario tiene que imitar
    """
    imagen = cv2.imread(str(ruta_imagen))

    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(ruta_modelo)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_poses=1,
    )

    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
        resultado = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)))

    if not resultado.pose_landmarks:
        raise ValueError("El modelo no ha detectado ninguna persona en la imagen.")

    return resultado, imagen.shape[1], imagen.shape[0]


def registrar_pose_desde_imagen(
    ruta_imagen, 
    ruta_salida_json, 
    nombre_pose, 
    articulaciones_relevantes, 
    tolerancia=10, 
    mantener_s=10,
    pesos=None
):
    """
    Mide los ángulos de la foto y los guarda como postura en un JSON junto con el esqueleto (para poder dibujarlo luego).
    Los ángulos que no son relevantes se guardan con todo a None
    """
    resultado, ancho, alto = detectar_pose(ruta_imagen, MODEL_PATH)
    puntos, visibilidad = extraer_puntos(resultado, ancho, alto)
    medidos = medir_angulos(puntos, ANGULOS, visibilidad)

    angulos = {}
    for clave, datos in ANGULOS["angulos"].items():
        relevante = clave in articulaciones_relevantes
        
        angulos[clave] = {
            "puntos": list(datos["puntos"]),
            "objetivo": round(float(medidos[clave]), 1) if relevante and medidos[clave] is not None else None,
            "tol": tolerancia if relevante else None,
            "peso": peso_de(clave, pesos) if relevante else None,
        }

    contenido = {
        "postura": {
            "nombre": nombre_pose,
            "mantener_s": mantener_s,
            "angulos": angulos,
        },
        "esqueleto": esqueleto_a_dict(resultado, ancho, alto),
    }

    salida = Path(ruta_salida_json)
    salida.parent.mkdir(parents=True, exist_ok=True)
    with salida.open("w", encoding="utf-8") as f:
        json.dump(contenido, f, indent=2, ensure_ascii=False)

    return contenido