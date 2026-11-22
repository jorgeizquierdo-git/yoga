"""
Interfaz gráfica para el juego. Posiblemente sea un código corto, pero podemos aprovechar (en caso de tener tiempo), de añadir funcionalidades de ayuda
tales como "puntos de apoyo" (como guía para las posturas) o seleccionar con posturas (creo que es más original) las diferentes dificultades del juego.

dibujar(frame, estado, result) pinta encima del fotograma de la cámara lo que toque según la fase del juego
(estado = lo que devuelve Juego.actualizar). Todo se dibuja con OpenCV, sin tildes (no las dibuja).
"""
import cv2
import numpy as np

import game
from tracker_poses import dibujar_esqueleto
from utils import UMBRAL_APROBADO, VENTANA_PUNTUACION

BLANCO, NEGRO = (255, 255, 255), (0, 0, 0)
VERDE, ROJO, AMARILLO, GRIS = (0, 220, 0), (0, 0, 255), (0, 220, 255), (150, 150, 150)
ESQUINAS_MENU = {"facil": "arriba_izq", "medio": "arriba_dcha", "dificil": "abajo_dcha"}   # abajo_izq: salir
TAMANO_MINIATURA = 0.22      # ancho de las miniaturas (fracción del ancho de la imagen)

_cache_imagenes = {}


def _imagen(ruta):
    if ruta not in _cache_imagenes:
        _cache_imagenes[ruta] = cv2.imread(ruta) if ruta else None
    return _cache_imagenes[ruta]


def _escala(frame):
    return frame.shape[1] / 640.0


def texto(frame, txt, x, y, color=BLANCO, tam=0.7, grosor=2, centrado=False):
    """Texto con contorno negro para que se lea sobre cualquier fondo. tam es relativo a 640 px de ancho."""
    tam *= _escala(frame)
    grosor = max(1, int(round(grosor * _escala(frame))))
    if centrado:
        (w, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, tam, grosor)
        x = int(x - w / 2)
    cv2.putText(frame, txt, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, tam, NEGRO, grosor + 3, cv2.LINE_AA)
    cv2.putText(frame, txt, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, tam, color, grosor, cv2.LINE_AA)


def oscurecer(frame, cantidad=0.6):
    frame[:] = (frame * (1 - cantidad)).astype(np.uint8)


def pegar(frame, img, x, y, ancho, borde=None):
    """Pega `img` reescalada a `ancho` con su esquina superior izquierda en (x, y). Devuelve (x, y, w, h)."""
    if img is None:
        return x, y, 0, 0
    h = int(img.shape[0] * ancho / img.shape[1])
    H, W = frame.shape[:2]
    h, ancho = min(h, H - y), min(ancho, W - x)
    if h <= 0 or ancho <= 0:
        return x, y, 0, 0
    frame[y:y + h, x:x + ancho] = cv2.resize(img, (ancho, int(img.shape[0] * ancho / img.shape[1])))[:h, :ancho]
    if borde is not None:
        cv2.rectangle(frame, (x, y), (x + ancho - 1, y + h - 1), borde, max(2, int(3 * _escala(frame))))
    return x, y, ancho, h


def barra(frame, x, y, w, h, progreso, color=VERDE):
    cv2.rectangle(frame, (x, y), (x + w, y + h), NEGRO, -1)
    cv2.rectangle(frame, (x, y), (x + int(w * max(0.0, min(1.0, progreso))), y + h), color, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), BLANCO, 1)


def marco_rojo(frame, nadie=True):
    """Marco rojo alrededor de toda la imagen: no se detecta al jugador (nadie=True) o solo en parte."""
    H, W = frame.shape[:2]
    g = max(8, int(12 * _escala(frame)))
    cv2.rectangle(frame, (0, 0), (W - 1, H - 1), ROJO, g)
    if nadie:
        # En el centro: cuando sale este aviso no hay nadie en pantalla, así que no tapa al jugador
        texto(frame, "No se te ve: colocate delante de la camara", W / 2, H / 2, ROJO, 0.7, centrado=True)
    else:
        # Se le ve a medias: el aviso va arriba para no taparle
        texto(frame, "No se te ve entero: alejate o centrate", W / 2, H * 0.22, ROJO, 0.6, centrado=True)


def _color_nota(nota):
    return VERDE if nota >= UMBRAL_APROBADO else (AMARILLO if nota >= UMBRAL_APROBADO - 20 else ROJO)


# --- pantallas -------------------------------------------------------------------------------------
def _menu(frame, estado):
    H, W = frame.shape[:2]
    m = int(10 * _escala(frame))
    ancho = int(W * TAMANO_MINIATURA)
    menu = estado["menu"]
    for dificultad, esquina in ESQUINAS_MENU.items():
        img = _imagen(game._imagen_de(dificultad, "default", "media"))
        h = int(img.shape[0] * ancho / img.shape[1]) if img is not None else ancho
        x = m if esquina.endswith("izq") else W - ancho - m
        y = m + int(22 * _escala(frame)) if esquina.startswith("arriba") else H - h - m - int(22 * _escala(frame))
        elegida = dificultad == menu["candidata"]
        _, _, w, h = pegar(frame, img, x, y, ancho, borde=VERDE if elegida else BLANCO)
        punt = menu["puntuaciones"].get(dificultad)
        etiqueta = dificultad.upper() + ("" if punt is None else f"  {punt:.0f}")
        texto(frame, etiqueta, x, y - int(6 * _escala(frame)), VERDE if elegida else BLANCO, 0.55)
        if elegida:
            barra(frame, x, y + h + int(4 * _escala(frame)), w, int(10 * _escala(frame)), menu["progreso"])
    texto(frame, "Q: salir", m, H - m - int(8 * _escala(frame)), BLANCO, 0.6)
    texto(frame, "Haz la postura de una esquina para elegir", W / 2, H - m - int(40 * _escala(frame)),
          BLANCO, 0.55, centrado=True)


def _explicacion(frame, estado):
    H, W = frame.shape[:2]
    oscurecer(frame, 0.75)
    pose = estado["pose"]
    texto(frame, f"Postura {estado['indice'] + 1} de {estado['total']}: {pose['nombre']}", W / 2,
          int(40 * _escala(frame)), BLANCO, 0.8, centrado=True)
    img = _imagen(pose["imagen"])
    if img is not None:
        alto_max, ancho_max = int(H * 0.7), int(W * 0.8)
        ancho = min(ancho_max, int(img.shape[1] * alto_max / img.shape[0]))
        h = int(img.shape[0] * ancho / img.shape[1])
        pegar(frame, img, (W - ancho) // 2, int(60 * _escala(frame)) + (alto_max - h) // 2, ancho, borde=BLANCO)
    texto(frame, f"Fijate bien... {estado['restante']:.0f}", W / 2, H - int(20 * _escala(frame)),
          AMARILLO, 0.7, centrado=True)


def _guia(frame, estado):
    """Miniatura de la pose en la esquina superior derecha + nombre."""
    H, W = frame.shape[:2]
    m = int(10 * _escala(frame))
    ancho = int(W * TAMANO_MINIATURA)
    pegar(frame, _imagen(estado["pose"]["imagen"]), W - ancho - m, m, ancho, borde=BLANCO)
    texto(frame, f"{estado['indice'] + 1}/{estado['total']} {estado['pose']['nombre']}", m,
          int(28 * _escala(frame)), BLANCO, 0.6)


def _ejecucion(frame, estado):
    H, W = frame.shape[:2]
    _guia(frame, estado)
    m = int(10 * _escala(frame))
    color = ROJO if estado["restante"] <= 5 else BLANCO           # se pone rojo en los últimos 5 s
    texto(frame, f"Quedan {estado['restante']:.0f} s", m, int(58 * _escala(frame)), color, 0.7)
    comp = estado["comparacion"]
    if comp and comp["puntuacion"] is not None:
        texto(frame, f"{comp['puntuacion']:.0f}", m, int(95 * _escala(frame)), _color_nota(comp["puntuacion"]), 1.0)
    progreso = estado["mantenido"] / VENTANA_PUNTUACION
    ancho_barra = int(W * 0.5)
    texto(frame, "Manten la postura", W / 2, H - int(42 * _escala(frame)), BLANCO, 0.55, centrado=True)
    barra(frame, (W - ancho_barra) // 2, H - int(34 * _escala(frame)), ancho_barra, int(14 * _escala(frame)), progreso)


def _nota_pose(frame, estado):
    H, W = frame.shape[:2]
    oscurecer(frame, 0.6)
    nota = estado["nota"]
    texto(frame, estado["pose"]["nombre"], W / 2, H / 2 - int(40 * _escala(frame)), BLANCO, 0.9, centrado=True)
    texto(frame, f"{nota:.0f}", W / 2, H / 2 + int(30 * _escala(frame)), _color_nota(nota), 2.2, 4, centrado=True)


def _resultados(frame, estado):
    H, W = frame.shape[:2]
    oscurecer(frame, 0.8)
    e = _escala(frame)
    texto(frame, f"Resultados - {str(estado['dificultad']).upper()}", W / 2, int(45 * e), BLANCO, 0.9, centrado=True)
    y = int(95 * e)
    for r in estado["resultados"]:
        texto(frame, r["nombre"], W * 0.18, y, BLANCO, 0.65)
        texto(frame, f"{r['nota']:.0f}", W * 0.75, y, _color_nota(r["nota"]), 0.75)
        y += int(34 * e)
    cv2.line(frame, (int(W * 0.15), y - int(18 * e)), (int(W * 0.85), y - int(18 * e)), BLANCO, 1)
    media = estado.get("media", 0.0)
    texto(frame, "Media", W * 0.18, y + int(12 * e), BLANCO, 0.8)
    texto(frame, f"{media:.0f}", W * 0.75, y + int(12 * e), _color_nota(media), 1.0)
    texto(frame, "ENTER: volver al menu     Q: salir", W / 2, H - int(20 * e), GRIS, 0.55, centrado=True)


PANTALLAS = {game.MENU: _menu, game.EXPLICACION: _explicacion, game.EJECUCION: _ejecucion, game.NOTA_POSE: _nota_pose, game.RESULTADOS: _resultados}


def dibujar(frame, estado, result):
    """Pinta el esqueleto (coloreado si hay pose que comparar) y la pantalla de la fase actual."""
    if result.pose_landmarks and estado["fase"] in (game.MENU, game.EJECUCION):
        dibujar_esqueleto(frame, result.pose_landmarks[0], None, estado["comparacion"])
    PANTALLAS[estado["fase"]](frame, estado)
    if estado["fase"] in (game.NOTA_POSE, game.RESULTADOS):
        return
    if not estado["detectado"]:
        marco_rojo(frame, nadie=True)
    elif not estado.get("visible", True):
        marco_rojo(frame, nadie=False)