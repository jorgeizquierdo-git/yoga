import time
import cv2
import mediapipe as mp
import ui
from game import Juego
from sonidos import Sonidos
from suavizado import FiltroEsqueleto
from utils import MODEL_PATH, CAMERA_INDEX

def main():
    """
    Funcion principal de TODO lo que hemos hecho. Muy similar al main de tracker_poses.py
    """
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("No se pudo abrir la cámara, prueba con otro CAMERA_INDEX")
        return

    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=1,
    )

    juego, sonidos, filtro = Juego(), Sonidos(), FiltroEsqueleto()
    cv2.namedWindow("Yoga", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    escalar = True
    t0 = time.monotonic()

    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            
            if not ok:
                print("No se pudo leer un frame de la cámara :(")
                break

            frame = cv2.flip(frame, 1)
            alto, ancho = frame.shape[0], frame.shape[1]
            if escalar:
                cv2.resizeWindow("Yoga", ancho * 2, alto * 2)
                escalar = False

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            ahora = time.monotonic()
            result = filtro.aplicar(landmarker.detect_for_video(mp_image, int((ahora - t0) * 1000)), ahora)

            estado = juego.actualizar(result, ancho, alto, ahora)
            sonidos.reproducir_eventos(estado["eventos"])
            ui.dibujar(frame, estado, result)
            cv2.imshow("Yoga", frame)

            tecla = cv2.waitKey(1) & 0xFF
            if tecla in (27, ord("q")):
                break

            if tecla == ord("m") or (estado["fase"] == "resultados" and tecla in (13, 32)):
                juego.ir_al_menu()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()