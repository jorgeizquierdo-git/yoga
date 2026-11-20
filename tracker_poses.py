import os
import time

import cv2
import mediapipe as mp

from utils import (MIN_VISIBILIDAD, CAMERA_INDEX, MODEL_PATH, NOMBRES, ANGULOS, COLOR_HUESO, COLOR_PUNTO_CORRECTO, COLOR_PUNTO_INCORRECTO, COLOR_PUNTO_PREDETERMINADO,
INTERVALOS_PRINT, INTERVALOS_COMPROBACION, extraer_puntos, medir_angulos, _indice, comprobar_encuadre, puntos_necesarios)


BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

def cargar_conexiones():
    try:
        conexiones = mp.tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS
        return [(c.start, c.end) for c in conexiones]
    except AttributeError:
        return list(mp.solutions.pose.POSE_CONNECTIONS)

CONEXIONES_POSICION = cargar_conexiones()

def dibujar_esqueleto(frame, landmarks):
    h, w = frame.shape[:2]
    puntos = [(int(p.x * w), int(p.y * h)) for p in landmarks]
    visibles = [p.visibility >= MIN_VISIBILIDAD for p in landmarks]

    for i, j in CONEXIONES_POSICION:
        if visibles[i] and visibles[j]:
            cv2.line(frame, puntos[i], puntos[j], COLOR_HUESO, 2)

def imprimir_coordenadas(landmarks, world_landmarks, t_seg):
    print(f"Instante t={t_seg}")
    print(f"{'idx':>3} {'nombre':<14} {'x':>7} {'y':>7} {'z':>7} {'vis':>5} | "f"{'wx(m)':>7} {'wy(m)':>7} {'wz(m)':>7}")

    for i, punto in enumerate(landmarks):
        wp = world_landmarks[i]
        mundo = f"{wp.x:7.3f} {wp.y:7.3f} {wp.z:7.3f}"
        puntos = f"{punto.x:7.3f} {punto.y:7.3f} {punto.z:7.3f} {punto.visibility:5.2f}"
        print(f"{i:>3}{NOMBRES[i]:<14} {puntos} | {mundo}")        

def imprimir_angulos(result):
    print(f"Angulos con puntos 'WORLD', visibilidad minima = {MIN_VISIBILIDAD}")

    puntos, visibilidad = extraer_puntos(result)
    if puntos is None:
        print("No hay puntos para calcular ángulos")
        return

    medidos = medir_angulos(puntos, ANGULOS, visibilidad)
    print(f"{'angulo':<20} {'puntos (vertice en el medio)':<30} {'vis.min':>7} {'grados':>13}")
    for clave, datos in ANGULOS["angulos"].items():
        ids = [_indice(p) for p in datos["puntos"]]
        vmin = min(visibilidad[i] for i in ids)
        valor = medidos[clave]
        texto = f"{valor:13.1f}" if valor is not None else f"{'no evaluable':>13}"
        print(f"{clave:<20} {'-'.join(map(str, ids)):<30} {vmin:7.2f} {texto}")

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"El camino {MODEL_PATH} no lleva a ningún modelo")
        return

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("No se pudo o no se encontro ninguna cámara.")
        return

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
    )

    t0 = time.monotonic()
    t_prev = t0
    intervalo_print = t0

    with PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("No se pudo leer un frame de la cámara :(")
                return

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            ahora = time.monotonic()
            timestamp_ms = int((ahora - t0) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            tecla = cv2.waitKey(1) & 0xFF
            if tecla in (27, ord("q")):
                break

            if not result.pose_landmarks:
                cv2.putText(frame, "No se detecta ninguna persona", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                continue
            
            landmarks = result.pose_landmarks[0]
            dibujar_esqueleto(frame, landmarks)
            """
            ok_encuadre, avisos = comprobar_encuadre(result)
            if ok_encuadre:
                cv2.putText(frame, "Encuadre OK", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                for n, aviso in enumerate(avisos):
                    cv2.putText(frame, aviso, (10, 60 + 28 * n), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            """
            if ahora - intervalo_print >= INTERVALOS_PRINT:
                world_landmarks = result.pose_world_landmarks[0] if result.pose_world_landmarks else None
                imprimir_coordenadas(landmarks, world_landmarks, ahora - intervalo_print)
                imprimir_angulos(result)

            if ahora - intervalo_print >= INTERVALOS_COMPROBACION:
                pass #Aqui hay que implementar la comprobación de la postura del usuario con uno concreto (inyectado directamente a la función). Preferimos la imagen p3

            fps = 1.0 / max(ahora - t_prev, 1e-6)
            t_prev = ahora
            cv2.putText(frame, f"FPS: {fps:.0f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Esqueleto", frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()