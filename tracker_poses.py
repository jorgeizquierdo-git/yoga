import os
import time
import cv2
import mediapipe as mp
from utils import (
    MIN_VISIBILIDAD, CAMERA_INDEX, MODEL_PATH, NOMBRES, ANGULOS, MARGEN_CERCA, NOMBRES_EN_PANTALLA, COLOR_HUESO, COLOR_TEXTO, 
    COLOR_PUNTO_PREDETERMINADO, COLOR_PUNTO_CORRECTO, COLOR_PUNTO_CERCA, COLOR_PUNTO_INCORRECTO, INTERVALOS_COMPROBACION, 
    extraer_puntos, medir_angulos, indice
)
from poses import comparar_pose, cargar_postura
from suavizado import FiltroEsqueleto

RUTA_POSE = "jason/medio/default.json"
# revisar medio p4, dificil p1

CONEXIONES = [(c.start, c.end) for c in mp.tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS]
GRAVEDAD = {
    COLOR_PUNTO_CORRECTO: 0, 
    COLOR_PUNTO_CERCA: 1, 
    COLOR_PUNTO_INCORRECTO: 2
}


def escribir(frame, texto, x, y, color=COLOR_TEXTO):
    """
    Texto con contorno negro para que se lea sobre cualquier fondo
    """
    cv2.putText(frame, texto, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, texto, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return None


def color_angulo(error, tol):
    """
    Verde si está dentro de la tolerancia, amarillo si se pasa menos de MARGEN_CERCA y rojo si se pasa más o no se pudo medir
    """
    if error is None:
        return COLOR_PUNTO_INCORRECTO
    
    if abs(error) <= tol:
        return COLOR_PUNTO_CORRECTO
    
    if abs(error) <= tol + MARGEN_CERCA:
        return COLOR_PUNTO_CERCA
    
    return COLOR_PUNTO_INCORRECTO


def colores_comparacion(comparacion):
    """
    Devuelve una tupla de color del ángulo y del vértice. En los casos donde se calculan dos agnulos en el mismo
    vertice se deja el pero color para no ocultar el fallo.
    """
    por_angulo, por_vertice = {}, {}

    if not comparacion:
        return por_angulo, por_vertice

    for clave, d in comparacion["detalle"].items():
        color = color_angulo(d["error"], d["tol"])
        por_angulo[clave] = color
        vertice = indice(ANGULOS["angulos"][clave]["puntos"][1]) # Accede al punto de cálculo del vértice "clave"
        
        if vertice not in por_vertice or GRAVEDAD[color] > GRAVEDAD[por_vertice[vertice]]: # Aquí se empeora el estado si existe otro angulo peor
            por_vertice[vertice] = color

    return por_angulo, por_vertice


def etiquetas_por_vertice(medidos):
    """
    Textos de los ángulos agrupados por vertice. El nombre del vertice solo se pone en las articulaciones NOMBRES_EN_PANTALLA
    """

    etiquetas = {}
    for clave, datos in ANGULOS["angulos"].items():
        vertice = indice(datos["puntos"][1])
        valor = "--" if medidos.get(clave) is None else f"{medidos[clave]:.0f}"
        
        tipo = clave.rsplit("_", 1)[0]
        nombre = NOMBRES[vertice] if tipo in NOMBRES_EN_PANTALLA else tipo.replace("_", "-")
        
        etiquetas.setdefault(vertice, []).append((clave, f"{nombre} {valor}"))
    
    return etiquetas


def dibujar_esqueleto(frame, landmarks, medidos=None, comparacion=None):
    """
    Dibuja huesos y puntos. No existe mejor explicacion
    """
    alto, ancho = frame.shape[0], frame.shape[1]

    puntos = [(int(punto.x * ancho), int(punto.y * alto)) for punto in landmarks]
    visibles = [punto.visibility >= MIN_VISIBILIDAD for punto in landmarks]

    for i, j in CONEXIONES:
        if visibles[i] and visibles[j]:
            cv2.line(frame, puntos[i], puntos[j], COLOR_HUESO, 2)

    color_por_angulo, color_por_vertice = colores_comparacion(comparacion)
    for i, punto in enumerate(puntos):
        if visibles[i]:
            cv2.circle(frame, punto, 4, color_por_vertice.get(i, COLOR_PUNTO_PREDETERMINADO), -1)

    if medidos is None:
        return

    for vertice, lineas in etiquetas_por_vertice(medidos).items():
        if visibles[vertice]:
            x, y = puntos[vertice]
            for n, (clave, texto) in enumerate(lineas):
                escribir(frame, texto, x + 8, y - 6 + 16 * n, color_por_angulo.get(clave, COLOR_TEXTO))


def imprimir_comparacion(res):
    """
    Exactametente el mismo motivo que la función de arriba
    """
    if res is None:
        print(f"Sin comparacion: no existe la pose {RUTA_POSE}")
        return

    puntuacion = "-" if res["puntuacion"] is None else f"{res['puntuacion']:.1f}"
    cobertura = f"{100 * res.get('cobertura', 0):.0f}%"
    print(f"Comparacion con la pose: estado={res['estado']} puntuacion={puntuacion} medido={cobertura} del peso lado={res.get('lado', '-')}")
    for aviso in res["avisos"]:
        print(f"  ! {aviso}")

    for clave, d in res["detalle"].items():
        observado = "-" if d["observado"] is None else f"{d['observado']:.1f}"
        error = "-" if d["error"] is None else f"{d['error']:+.1f}"
        nota = "-" if d["nota"] is None else f"{d['nota']:.2f}"
        print(f"  {clave:<20} observado={observado:>6} objetivo={d['objetivo']:>6} error={error:>7} nota={nota} peso={d.get('peso', 1):g}")

def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("No se pudo o no se encontro ninguna cámara.")
        return

    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=1,
    )

    pose = cargar_postura(RUTA_POSE) if os.path.exists(RUTA_POSE) else None
    if pose is None:
        print(f"Joder Jorge, no me has avisado que no he puesto una postura")

    filtro = FiltroEsqueleto()
    t0 = ultima_comprobacion = time.monotonic()
    cv2.namedWindow("Esqueleto", cv2.WINDOW_NORMAL or cv2.WINDOW_KEEPRATIO)
    escalar = True

    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("No se pudo leer un frame de la cámara :(")
                break

            frame = cv2.flip(frame, 1)
            alto, ancho = frame.shape[0], frame.shape[1]
            if escalar:
                cv2.resizeWindow("Esqueleto", ancho * 2, alto * 2)
                escalar = False
            
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            ahora = time.monotonic()
            result = filtro.aplicar(landmarker.detect_for_video(mp_image, int((ahora - t0) * 1000)), ahora)

            if (cv2.waitKey(1) & 0xFF) in (27, ord("q")): # Salir con "q", aunque me gustaria que fuera con alt + f4
                break

            if not result.pose_landmarks:
                cv2.putText(frame, "No se detecta ninguna persona", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow("Esqueleto", frame)
                continue

            landmarks = result.pose_landmarks[0] # puntos normales (no los WORLD basura)
            puntos, visibilidad = extraer_puntos(result, ancho, alto)
            medidos = medir_angulos(puntos, ANGULOS, visibilidad)
            comparacion = comparar_pose(result, pose, ancho, alto) if pose else None
            dibujar_esqueleto(frame, landmarks, medidos, comparacion)

            if ahora - ultima_comprobacion >= INTERVALOS_COMPROBACION:
                imprimir_comparacion(comparacion)
                ultima_comprobacion = ahora
            
            cv2.imshow("Esqueleto", frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()