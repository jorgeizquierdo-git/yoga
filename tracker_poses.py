import os
import time
from datetime import datetime

import cv2
import mediapipe as mp

from utils import (MIN_VISIBILIDAD, CAMERA_INDEX, MODEL_PATH, NOMBRES, ANGULOS, COLOR_HUESO, COLOR_PUNTO_CORRECTO, COLOR_PUNTO_INCORRECTO, COLOR_PUNTO_PREDETERMINADO,
COLOR_PUNTO_CERCA, MARGEN_CERCA, ESCALA_VENTANA,
COLOR_TEXTO, INTERVALOS_PRINT, INTERVALOS_COMPROBACION, INTERVALO_CAPTURA, CARPETA_CAPTURAS, NOMBRES_EN_PANTALLA,
extraer_puntos, medir_angulos, _indice)
from poses import comparar_pose, cargar_postura
from registrador_poses import registrar_pose_desde_imagen
from suavizado import FiltroEsqueleto


RUTA_POSE = "jason/dificil/p4.json"     # pose con la que se compara en vivo (colores verde/rojo)
# revisar medio p4, dificil mirar p1
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


def _texto(frame, texto, x, y, color=COLOR_TEXTO, escala=0.45):
    """putText con contorno negro, para que se lea sobre cualquier fondo."""
    cv2.putText(frame, texto, (x, y), cv2.FONT_HERSHEY_SIMPLEX, escala, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, texto, (x, y), cv2.FONT_HERSHEY_SIMPLEX, escala, color, 1, cv2.LINE_AA)


def _etiquetas_por_vertice(medidos):
    """
    Agrupa los ángulos de ANGULOS por su vértice (el punto del medio) -> {indice: [lineas de texto]}.
    En el hombro coinciden dos ángulos ("hombro_izq" y "cabeza_tronco_izq"), por eso es una lista.
    """
    etiquetas = {}
    for clave, datos in ANGULOS["angulos"].items():
        vertice = _indice(datos["puntos"][1])
        valor = medidos.get(clave)
        valor_txt = "--" if valor is None else f"{valor:.0f}"
        tipo = clave.rsplit("_", 1)[0]                     # "rodilla_izq" -> "rodilla"
        if tipo in NOMBRES_EN_PANTALLA:
            linea = f"{NOMBRES[vertice]} {valor_txt}"      # p. ej. "rodilla_der 172"
        else:
            linea = f"{tipo.replace('_', '-')} {valor_txt}"  # p. ej. "cabeza-tronco 131": sin nombre del punto
        etiquetas.setdefault(vertice, []).append((clave, linea))
    return etiquetas


def color_angulo(error, tol):
    """
    Verde si |error| <= tol, amarillo si se pasa por menos de MARGEN_CERCA grados, rojo si se pasa más
    o si el ángulo no se pudo medir (error None).
    """
    if error is None:
        return COLOR_PUNTO_INCORRECTO
    if abs(error) <= tol:
        return COLOR_PUNTO_CORRECTO
    if abs(error) <= tol + MARGEN_CERCA:
        return COLOR_PUNTO_CERCA
    return COLOR_PUNTO_INCORRECTO


def _colores_por_angulo(comparacion):
    """{clave del ángulo: color} para los ángulos que evalúa la pose (el texto de cada ángulo va de su color)."""
    if not comparacion:
        return {}
    return {clave: color_angulo(d["error"], d["tol"] if d.get("tol") is not None else 10)
            for clave, d in comparacion["detalle"].items()}


def _colores_por_vertice(comparacion):
    """
    {indice del vértice: color} para los ángulos que evalúa la pose. Los vértices que no aparecen aquí
    (ángulos no relevantes, manos, cara...) se pintan con COLOR_PUNTO_PREDETERMINADO.
    Si en un vértice coinciden dos ángulos evaluados (el hombro: "hombro_x" y "cabeza_tronco_x"),
    se queda el peor color, para no ocultar un fallo.
    """
    gravedad = {COLOR_PUNTO_CORRECTO: 0, COLOR_PUNTO_CERCA: 1, COLOR_PUNTO_INCORRECTO: 2}
    colores = {}
    if not comparacion:
        return colores
    for clave, d in comparacion["detalle"].items():
        datos = ANGULOS["angulos"].get(clave)
        if datos is None:
            continue
        vertice = _indice(datos["puntos"][1])
        tol = d["tol"] if d.get("tol") is not None else 10
        color = color_angulo(d["error"], tol)
        if vertice not in colores or gravedad[color] > gravedad[colores[vertice]]:
            colores[vertice] = color
    return colores


def dibujar_esqueleto(frame, landmarks, medidos=None, comparacion=None):
    """
    Dibuja huesos, puntos y, si se pasan `medidos` ({clave: grados} de ANGULOS), el valor de cada ángulo
    junto a su vértice. El nombre del punto solo se escribe en las articulaciones de NOMBRES_EN_PANTALLA.
    Si se pasa `comparacion` (salida de comparar_pose), los vértices evaluados salen en verde o rojo.
    """
    h, w = frame.shape[:2]
    puntos = [(int(p.x * w), int(p.y * h)) for p in landmarks]
    visibles = [p.visibility >= MIN_VISIBILIDAD for p in landmarks]

    for i, j in CONEXIONES_POSICION:
        if visibles[i] and visibles[j]:
            cv2.line(frame, puntos[i], puntos[j], COLOR_HUESO, 2)

    colores = _colores_por_vertice(comparacion)
    colores_angulo = _colores_por_angulo(comparacion)
    for k, (x, y) in enumerate(puntos):
        if visibles[k]:
            cv2.circle(frame, (x, y), 4, colores.get(k, COLOR_PUNTO_PREDETERMINADO), -1)

    if medidos is None:
        return
    for vertice, lineas in _etiquetas_por_vertice(medidos).items():
        if not visibles[vertice]:
            continue
        x, y = puntos[vertice]
        for n, (clave, linea) in enumerate(lineas):
            _texto(frame, linea, x + 8, y - 6 + 16 * n, colores_angulo.get(clave, COLOR_TEXTO))


def imprimir_coordenadas(landmarks, ancho, alto, t_seg):
    """Coordenadas 2D de los 33 puntos: normalizadas (0-1) y en píxeles, que son las que se usan."""
    print(f"Instante t={t_seg:.1f}s  (imagen {ancho}x{alto})")
    print(f"{'idx':>3} {'nombre':<14} {'x':>7} {'y':>7} {'vis':>5} | {'x(px)':>7} {'y(px)':>7}")

    for i, punto in enumerate(landmarks):
        normalizado = f"{punto.x:7.3f} {punto.y:7.3f} {punto.visibility:5.2f}"
        pixeles = f"{punto.x * ancho:7.1f} {punto.y * alto:7.1f}"
        print(f"{i:>3} {NOMBRES[i]:<14} {normalizado} | {pixeles}")

def imprimir_angulos(result, ancho, alto):
    print(f"Angulos 2D (x, y en pixeles), visibilidad minima = {MIN_VISIBILIDAD}")

    puntos, visibilidad = extraer_puntos(result, ancho, alto)
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

def imprimir_comparacion(res):
    if res is None:
        print(f"Sin comparacion: no existe la pose {RUTA_POSE}")
        return
    puntuacion = "-" if res["puntuacion"] is None else f"{res['puntuacion']:.1f}"
    cobertura = f"{100 * res.get('cobertura', 0):.0f}%"
    print(f"Comparacion con la pose: estado={res['estado']} puntuacion={puntuacion} "
          f"medido={cobertura} del peso lado={res.get('lado', '-')}")

    for aviso in res["avisos"]:
        print(f"  ! {aviso}")

    for clave, d in res["detalle"].items():
        observado = "-" if d["observado"] is None else f"{d['observado']:.1f}"
        error = "-" if d["error"] is None else f"{d['error']:+.1f}"
        nota = "-" if d["nota"] is None else f"{d['nota']:.2f}"
        print(f"  {clave:<20} observado={observado:>6} objetivo={d['objetivo']:>6} error={error:>7} "
              f"nota={nota} peso={d.get('peso', 1):g}")


def capturar_esqueleto(frame_bgr):
    """
    Pasa un frame de la cámara (en memoria, sin guardarlo como imagen) por el mismo camino que las fotos
    (registrar_pose_desde_imagen) y guarda el resultado en CARPETA_CAPTURAS/captura_HHMMSS.json.
    Se marcan los 12 ángulos de ANGULOS como relevantes para poder analizarlos todos.
    Devuelve la ruta del .json, o None si no se detectó a nadie.
    """
    marca = datetime.now().strftime("%H%M%S")
    ruta = os.path.join(CARPETA_CAPTURAS, f"captura_{marca}.json")
    try:
        contenido = registrar_pose_desde_imagen(
            ruta_imagen=frame_bgr,
            ruta_modelo=MODEL_PATH,
            ruta_salida_json=ruta,
            nombre_pose=f"captura_{marca}",
            articulaciones_relevantes=list(ANGULOS["angulos"].keys()),
            tolerancia=10,
        )
    except ValueError as e:                 # no se detectó a nadie en el frame
        print(f"[captura] No se guardó: {e}")
        return None

    print(f"[captura] Guardada en {ruta} (angulos + {len(contenido['esqueleto']['puntos'])} puntos del esqueleto)")
    for clave, d in contenido["postura"]["angulos"].items():
        valor = "no evaluable" if d["objetivo"] is None else f"{d['objetivo']:.1f}"
        print(f"  {clave:<20} {valor}")
    return ruta


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

    # Se carga una vez, no en cada frame. Si no existe (p. ej. aún no se han generado los .json), el tracker
    # funciona igual pero sin comparar ni colorear.
    pose = cargar_postura(RUTA_POSE) if os.path.exists(RUTA_POSE) else None
    if pose is None:
        print(f"[aviso] No existe {RUTA_POSE}: se muestra el esqueleto sin comparar con ninguna pose")

    filtro = FiltroEsqueleto()
    t0 = time.monotonic()
    t_prev = t0
    ultimo_print = t0
    ultima_comprobacion = t0
    ultima_captura = t0
    ultimo_fps = t0

    # Ventana redimensionable (con el ratón) que se abre a ESCALA_VENTANA veces el tamaño de la cámara.
    # Se dibuja a tamaño de cámara y la ventana lo escala: los textos crecen igual que la imagen.
    cv2.namedWindow("Esqueleto", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    ventana_ajustada = False

    with PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("No se pudo leer un frame de la cámara :(")
                return

            frame_original = frame.copy()     # sin voltear: es el que se manda a la captura
            frame = cv2.flip(frame, 1)
            alto, ancho = frame.shape[:2]
            if not ventana_ajustada:
                cv2.resizeWindow("Esqueleto", ancho * ESCALA_VENTANA, alto * ESCALA_VENTANA)
                ventana_ajustada = True
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            ahora = time.monotonic()
            timestamp_ms = int((ahora - t0) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            result = filtro.aplicar(result, ahora)      # quita temblores y saltos (ver suavizado.py)

            tecla = cv2.waitKey(1) & 0xFF
            if tecla in (27, ord("q")):
                break

            # Captura periódica (aunque en este frame no se detecte a nadie: lo decide la propia función).
            # El resultado solo se avisa por consola, para no ensuciar la pantalla.
            if ahora - ultima_captura >= INTERVALO_CAPTURA:
                capturar_esqueleto(frame_original)
                ultima_captura = time.monotonic()   # tras la captura: crear el detector tarda un poco

            if not result.pose_landmarks:
                cv2.putText(frame, "No se detecta ninguna persona", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow("Esqueleto", frame)
                continue

            landmarks = result.pose_landmarks[0]
            puntos, visibilidad = extraer_puntos(result, ancho, alto)
            medidos = medir_angulos(puntos, ANGULOS, visibilidad)
            comparacion = comparar_pose(result, pose, ancho, alto) if pose else None
            dibujar_esqueleto(frame, landmarks, medidos, comparacion)

            if ahora - ultimo_print >= INTERVALOS_PRINT:
                imprimir_coordenadas(landmarks, ancho, alto, ahora - t0)
                imprimir_angulos(result, ancho, alto)
                ultimo_print = ahora

            if ahora - ultima_comprobacion >= INTERVALOS_COMPROBACION:
                imprimir_comparacion(comparacion)
                ultima_comprobacion = ahora

            # FPS, puntuación y lado van a consola (con la comparación), no a la pantalla
            fps = 1.0 / max(ahora - t_prev, 1e-6)
            t_prev = ahora
            if ahora - ultimo_fps >= INTERVALOS_COMPROBACION:
                print(f"FPS: {fps:.0f}")
                ultimo_fps = ahora
            cv2.imshow("Esqueleto", frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()