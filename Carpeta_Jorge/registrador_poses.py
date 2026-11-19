import json
from pathlib import Path
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from ..Carpeta_julio.utils import teta

ARTICULACIONS = {
    "codo_izq":[13,11,15],
    "codo_der":[14,12,16],
    "hombro_der":[11,13,23],
    "cadera_izq":[23,11,25],
    "cadera_der":[24,12,26],
    "rodilla_izq":[25,23,27],
    "rodilla_der":[26,24,28],
    "tobillo_izq":[27,25,31],
    "tobillo_der":[28,26,32]
}

def extraer_puntos(deteccion,ancho,alto):
    if deteccion.pose_world_landmarks:
        wl=deteccion.pose_world_landmarks[0]
        return np.array([[p.x,p.y,p.z] for p in wl],dtype=np.float64)
    
    if deteccion.pose_landmarks:
        pl = deteccion.pose_landmarks[0]
        
        return np.array(
            [[p.x*ancho,p.y*alto,p.z*ancho]for p in pl],
            dtype=np.float64
        )
        
    raise ValueError("El modelo no ha detectado a nadie o los ifs de la extraccion de puntos están mal")

def detectar_pose(ruta_imagen,ruta_modelo):
    imagen_bgr=cv2.imread(ruta_imagen)
    if imagen_bgr is None:
        raise FileNotFoundError(f'Nose ha podido encontrar la imagen {ruta_imagen}')
    
    alto, ancho, _ = imagen_bgr.shape
    
    imagen_rgb = cv2.cvtColor(imagen_bgr,cv2.COLOR_BGR2GRAY)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imagen_rgb)
    
    base_options=python.BaseOptions(model_asset_path=ruta_modelo)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1
    )
    
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        resultado = landmarker.detect(mp_image)
        
    puntos = extraer_puntos(resultado,ancho,alto)
    
    landmarks2d = resultado.pose_landmarks[0] if resultado.pose_landmarks else []
    return puntos, landmarks2d

def registrar_pose_desde_imagen(ruta_imagen,ruta_modelo,ruta_salida_json,nombre_pose,articulaciones_relevantes,tolerancia = 0):
    #articulaciones relevantes debe ser una lista de los string de nombre de la articulacion en articulacions
    puntos, landmarks_2d = detectar_pose(ruta_imagen='',ruta_modelo='') #ojito, cambiar.
    angulos_json={}
    for nombre,indices in ARTICULACIONS.items():
        idx_v,idx_1,idx_2 = indices
        if nombre in articulaciones_relevantes:
            objetivo = teta(puntos[idx_v],puntos[idx_1],puntos[idx_2])
            
        else:
            objetivo=None
        
        angulos_json[nombre] = {
            "puntos": indices,
            "objetivo": objetivo,
            "tolerancia": tolerancia
        }
        
    contenido = {
        "nombre": nombre,
        "imagen_origen": ruta_imagen,
        "angulos": angulos_json,
        "landmarks": [
            {
                "x": float(p.x),
                "y": float(p.y),
                "z": float(p.z),
                "visibility": float(getattr(p, "visibility", 0.0)),
            }
            for p in landmarks_2d
        ],
    }
    
    salida = Path(ruta_salida_json)
    salida.parent.mkdir(parents=True, exist_ok=True)
    with salida.open("w", encoding="utf-8") as f:
        json.dump(contenido,f,indent=2,ensure_ascii=False)
        
    return contenido

if __name__ == "__main__":
    registrar_pose_desde_imagen(
        ruta_imagen="",
        ruta_modelo="models/pose_landmarker_full.task",
        ruta_salida_json="",
        nombre_pose="",
        articulaciones_relevantes=[],
        tolerancia_defecto=0
    )
    
    
    
    
    