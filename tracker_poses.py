"""
Captura la cámara y dibuja el esqueleto (33 landmarks) con MediaPipe Pose Landmarker.

Uso (desde la carpeta del repo, con el entorno activado y los modelos descargados;
utils.py debe estar en la misma carpeta):
    python tracker_poses.py

Teclas:
    ESC o q -> salir
    i       -> cambia el modo de etiquetas: todos los índices / solo articulaciones clave / ninguno
    p       -> imprime coordenadas y ángulos por consola en ese mismo instante

En pantalla se avisa en rojo si no se ve el cuerpo entero (p. ej. "Retrocede: ...").

Cada PRINT_EVERY_S segundos se imprime por consola:
  1) Una tabla con la información de cada landmark:
    - x, y   : normalizados (0-1) respecto al ancho/alto de la imagen (x hacia la derecha, y hacia abajo)
    - z      : profundidad relativa (aprox. misma escala que x). Negativo = más cerca de la cámara
               que la cadera (el origen de z está en el punto medio de las caderas). Es la coordenada más ruidosa.
    - vis    : visibilidad (0-1), probabilidad de que el punto sea visible y no esté tapado
    - wx,wy,wz : coordenadas "world" en METROS, con origen en el centro de las caderas
                 (no dependen de la perspectiva de la cámara)
  2) El estado del encuadre (¿están dentro de la imagen y visibles los puntos necesarios?) y los
     ángulos de las articulaciones clave, calculados con utils.py (por defecto con las
     coordenadas world, que son las más fiables; se cambia con utils.USAR_WORLD).

MediaPipe NO tiene un landmark de "cuello": como referencia de la cabeza se usa la nariz (0).
"""
import os
import sys
import time

import cv2
import mediapipe as mp

import utils
from utils import NOMBRES, extraer_puntos, medir_angulos, comprobar_encuadre, puntos_necesarios

# ----------------------------------------------------------------------------
# Configuración global (la de los puntos/ángulos vive en utils.py)
# ----------------------------------------------------------------------------
MODEL_PATH = "models/pose_landmarker_full.task"
CAMERA_INDEX = 0
MIN_VISIBILITY = 0.1        # visibilidad mínima para DIBUJAR un punto (la de los ángulos está en utils.MIN_VISIBILIDAD)
PRINT_EVERY_S = 10          # cada cuántos segundos se vuelcan coordenadas y ángulos por consola

# Articulaciones en las que nos fijaremos (nariz como referencia de la cabeza, hombro, codo,
# muñeca, cadera, rodilla y talón).
INDICES_CLAVE = {0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 29, 30}

# Ángulos que se imprimen para comprobar que todo va bien. El vértice es el punto del medio.
# Mismo formato que las posturas del .json, sin "objetivo" ni "tol" porque aquí solo se mide.
ANGULOS_DEBUG = {"angulos": {
    "rodilla_izq":  {"puntos": ["cadera_izq", "rodilla_izq", "tobillo_izq"]},
    "rodilla_der":  {"puntos": ["cadera_der", "rodilla_der", "tobillo_der"]},
    "cadera_izq":   {"puntos": ["hombro_izq", "cadera_izq", "rodilla_izq"]},
    "cadera_der":   {"puntos": ["hombro_der", "cadera_der", "rodilla_der"]},
    "codo_izq":     {"puntos": ["hombro_izq", "codo_izq", "muneca_izq"]},
    "codo_der":     {"puntos": ["hombro_der", "codo_der", "muneca_der"]},
    "hombro_izq":   {"puntos": ["cadera_izq", "hombro_izq", "codo_izq"]},
    "hombro_der":   {"puntos": ["cadera_der", "hombro_der", "codo_der"]},
    "tobillo_izq":  {"puntos": ["rodilla_izq", "tobillo_izq", "punta_pie_izq"]},
    "tobillo_der":  {"puntos": ["rodilla_der", "tobillo_der", "punta_pie_der"]},
    "cabeza_tronco_izq": {"puntos": ["nariz", "hombro_izq", "cadera_izq"]},
    "cabeza_tronco_der": {"puntos": ["nariz", "hombro_der", "cadera_der"]},
}}

# Puntos que deben estar dentro de la imagen y bien visibles (los de los ángulos de arriba)
INDICES_ENCUADRE = puntos_necesarios(ANGULOS_DEBUG)

# Colores BGR
COLOR_HUESO = (255, 0, 0)
COLOR_PUNTO = (0, 255, 0)
COLOR_CLAVE = (0, 255, 255)       # amarillo

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


def _cargar_conexiones():
    """Pares (i, j) de landmarks unidos por un hueso. Compatible con mediapipe 0.10.x y 1.x."""
    try:                                    # mediapipe >= 1.0 (mp.solutions ya no existe)
        conexiones = mp.tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS
        return [(c.start, c.end) for c in conexiones]
    except AttributeError:                  # mediapipe 0.10.x
        return list(mp.solutions.pose.POSE_CONNECTIONS)


POSE_CONNECTIONS = _cargar_conexiones()


def texto_con_borde(frame, texto, pos, escala, color):
    """Texto con contorno negro para que se lea sobre cualquier fondo."""
    cv2.putText(frame, texto, pos, cv2.FONT_HERSHEY_SIMPLEX, escala, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, texto, pos, cv2.FONT_HERSHEY_SIMPLEX, escala, color, 1, cv2.LINE_AA)


def dibujar_esqueleto(frame, landmarks, modo_etiquetas=0):
    """
    Dibuja huesos, articulaciones e índices. landmarks: lista de 33 puntos normalizados (0-1).
    modo_etiquetas: 0 = todos los índices, 1 = solo articulaciones clave, 2 = ninguno.
    """
    h, w = frame.shape[:2]
    puntos = [(int(p.x * w), int(p.y * h)) for p in landmarks]
    visibles = [p.visibility >= MIN_VISIBILITY for p in landmarks]

    for i, j in POSE_CONNECTIONS:
        if visibles[i] and visibles[j]:
            cv2.line(frame, puntos[i], puntos[j], COLOR_HUESO, 2)

    for k, (x, y) in enumerate(puntos):
        if not visibles[k]:
            continue
        es_clave = k in INDICES_CLAVE
        cv2.circle(frame, (x, y), 6 if es_clave else 4, COLOR_CLAVE if es_clave else COLOR_PUNTO, -1)

        if modo_etiquetas == 0 or (modo_etiquetas == 1 and es_clave):
            texto_con_borde(frame, str(k), (x + 6, y - 6),
                            0.6 if es_clave else 0.4,
                            COLOR_CLAVE if es_clave else (255, 255, 255))


def imprimir_coordenadas(landmarks, world_landmarks, t_seg):
    """Vuelca por consola la información de cada landmark."""
    print(f"\n===== Coordenadas a los {t_seg:.1f} s =====")
    print(f"{'idx':>3} {'nombre':<14} {'x':>7} {'y':>7} {'z':>7} {'vis':>5} | "
          f"{'wx(m)':>7} {'wy(m)':>7} {'wz(m)':>7}")
    for k, p in enumerate(landmarks):
        marca = "*" if k in INDICES_CLAVE else " "
        if world_landmarks:
            wp = world_landmarks[k]
            mundo = f"{wp.x:7.3f} {wp.y:7.3f} {wp.z:7.3f}"
        else:
            mundo = "    n/d     n/d     n/d"
        print(f"{k:>3}{marca}{NOMBRES[k]:<14} {p.x:7.3f} {p.y:7.3f} {p.z:7.3f} {p.visibility:5.2f} | {mundo}")
    print("(* = articulación clave del juego)")


def imprimir_angulos(result):
    """Vuelca por consola el encuadre y los ángulos de ANGULOS_DEBUG calculados con utils.py."""
    sistema = "world (metros)" if utils.USAR_WORLD else "normalizadas (distorsionadas)"
    print(f"\n----- Ángulos [coordenadas {sistema}; visibilidad mínima {utils.MIN_VISIBILIDAD}] -----")

    ok_encuadre, avisos = comprobar_encuadre(result, INDICES_ENCUADRE)
    print("Encuadre: OK" if ok_encuadre else "Encuadre: " + " | ".join(avisos))

    puntos, visibilidad = extraer_puntos(result)
    if puntos is None:
        print("No hay puntos disponibles para calcular ángulos.")
        return
    medidos = medir_angulos(puntos, ANGULOS_DEBUG, visibilidad)
    print(f"{'angulo':<20} {'puntos (vertice en el medio)':<30} {'vis.min':>7} {'grados':>13}")
    for clave, datos in ANGULOS_DEBUG["angulos"].items():
        ids = [utils._indice(p) for p in datos["puntos"]]
        vmin = min(visibilidad[i] for i in ids)
        valor = medidos[clave]
        texto = f"{valor:13.1f}" if valor is not None else f"{'no evaluable':>13}"
        print(f"{clave:<20} {'-'.join(map(str, ids)):<30} {vmin:7.2f} {texto}")


def main():
    if not os.path.exists(MODEL_PATH):
        sys.exit(f"No se encuentra el modelo en {MODEL_PATH}.\nEjecuta antes: python download_models.py")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        sys.exit("No se pudo abrir la cámara. Prueba con otro CAMERA_INDEX o cierra otras apps que la usen.")

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
    )

    t0 = time.monotonic()
    t_prev = t0
    t_ultimo_print = t0
    modo_etiquetas = 0
    nombres_modo = ["todos los indices", "solo clave", "sin etiquetas"]

    with PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("No se pudo leer un frame de la cámara.")
                break

            frame = cv2.flip(frame, 1)                       # efecto espejo
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)     # OpenCV da BGR; MediaPipe espera RGB
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # Timestamp en ms, creciente y basado en el reloj real
            ahora = time.monotonic()
            timestamp_ms = int((ahora - t0) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            forzar_print = False
            tecla = cv2.waitKey(1) & 0xFF
            if tecla in (27, ord("q")):
                break
            elif tecla == ord("i"):
                modo_etiquetas = (modo_etiquetas + 1) % 3
            elif tecla == ord("p"):
                forzar_print = True

            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]
                dibujar_esqueleto(frame, landmarks, modo_etiquetas)

                # Aviso de encuadre en pantalla (ASCII: cv2.putText no dibuja tildes)
                ok_encuadre, avisos = comprobar_encuadre(result, INDICES_ENCUADRE)
                if ok_encuadre:
                    cv2.putText(frame, "Encuadre OK", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    for n, aviso in enumerate(avisos):
                        cv2.putText(frame, aviso, (10, 60 + 28 * n),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                if forzar_print or (ahora - t_ultimo_print) >= PRINT_EVERY_S:
                    world = result.pose_world_landmarks[0] if result.pose_world_landmarks else None
                    imprimir_coordenadas(landmarks, world, ahora - t0)
                    imprimir_angulos(result)
                    t_ultimo_print = ahora
            else:
                cv2.putText(frame, "No se detecta ninguna persona", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # FPS
            fps = 1.0 / max(ahora - t_prev, 1e-6)
            t_prev = ahora
            cv2.putText(frame, f"FPS: {fps:.0f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"[i] etiquetas: {nombres_modo[modo_etiquetas]}  [p] imprimir  [q] salir",
                        (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow("Esqueleto", frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()