"""Juego de posturas de yoga con MediaPipe Pose Landmarker.

Uso (desde la carpeta del repo, con el entorno activado, los modelos descargados y los .json generados):
    python main.py

Flujo: menú (haz la postura de una esquina para elegir dificultad) -> por cada postura de esa dificultad:
explicación, preparación y ejecución -> resultados con la nota de cada postura y la media.
La lógica está en game.py y el dibujo en ui.py; los tiempos se configuran en utils.py.

Teclas: ESC o q -> salir | m -> volver al menú | ENTER o espacio (en resultados) -> volver al menú
"""
import os
import sys
import time

import cv2
import mediapipe as mp

import ui
from game import Juego, RESULTADOS
from suavizado import FiltroEsqueleto
from utils import MODEL_PATH, CAMERA_INDEX, ESCALA_VENTANA

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode
VENTANA = "Yoga"


def main():
    if not os.path.exists(MODEL_PATH):
        sys.exit(f"No se encuentra el modelo en {MODEL_PATH}.\nEjecuta antes: python download_models.py")
    if not os.path.isdir("jason"):
        sys.exit("No existe la carpeta jason/. Genera antes las posturas: python regenerador_poses.py")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        sys.exit("No se pudo abrir la cámara. Prueba con otro CAMERA_INDEX o cierra otras apps que la usen.")

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
    )

    juego = Juego()
    filtro = FiltroEsqueleto()
    cv2.namedWindow(VENTANA, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    ventana_ajustada = False
    t0 = time.monotonic()

    with PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("No se pudo leer un frame de la cámara.")
                break

            frame = cv2.flip(frame, 1)                       # efecto espejo
            alto, ancho = frame.shape[:2]
            if not ventana_ajustada:
                cv2.resizeWindow(VENTANA, ancho * ESCALA_VENTANA, alto * ESCALA_VENTANA)
                ventana_ajustada = True

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)     # OpenCV da BGR; MediaPipe espera RGB
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ahora = time.monotonic()
            result = landmarker.detect_for_video(mp_image, int((ahora - t0) * 1000))
            result = filtro.aplicar(result, ahora)           # quita temblores y saltos (suavizado.py)

            estado = juego.actualizar(result, ancho, alto, ahora)
            ui.dibujar(frame, estado, result)
            cv2.imshow(VENTANA, frame)

            tecla = cv2.waitKey(1) & 0xFF
            if tecla in (27, ord("q")):
                break
            if tecla == ord("m") or (estado["fase"] == RESULTADOS and tecla in (13, 32)):
                juego.ir_al_menu()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()