"""Captura la cámara y dibuja el esqueleto (33 landmarks) con MediaPipe Pose Landmarker.

Uso (desde la carpeta del repo, con el entorno activado y los modelos descargados):
    python captura_esqueleto.py

Teclas: ESC o q -> salir
"""
import os
import sys
import time

import cv2
import mediapipe as mp

# Ruta del modelo
from utils import MODEL_PATH   # un único sitio para elegir el modelo
CAMERA_INDEX = 0 
MIN_VISIBILITY = 0.1 

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


def dibujar_esqueleto(frame, landmarks):
    """Dibuja huesos y articulaciones. landmarks: lista de 33 puntos normalizados (0-1)."""
    h, w = frame.shape[:2]
    puntos = [(int(p.x * w), int(p.y * h)) for p in landmarks]
    visibles = [p.visibility >= MIN_VISIBILITY for p in landmarks]

    for i, j in POSE_CONNECTIONS:
        if visibles[i] and visibles[j]:
            cv2.line(frame, puntos[i], puntos[j], (255, 0, 0), 2)
    for k, (x, y) in enumerate(puntos):
        if visibles[k]:
            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)


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
            timestamp_ms = int((time.monotonic() - t0) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.pose_landmarks:
                dibujar_esqueleto(frame, result.pose_landmarks[0])
            else:
                cv2.putText(frame, "No se detecta ninguna persona", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # FPS
            ahora = time.monotonic()
            fps = 1.0 / max(ahora - t_prev, 1e-6)
            t_prev = ahora
            cv2.putText(frame, f"FPS: {fps:.0f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("Esqueleto", frame)
            tecla = cv2.waitKey(1) & 0xFF
            if tecla in (27, ord("q")):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()