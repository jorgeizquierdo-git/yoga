import os
import cv2
import numpy as np
from tracker_poses import dibujar_esqueleto
from utils import UMBRAL_APROBADO, VENTANA_PUNTUACION

ESQUINAS_MENU = {"facil": "arriba_izq", "medio": "arriba_dcha", "dificil": "abajo_dcha"}
TAMANO_MINIATURA = 0.22 # Variable para ajustar las imágenes

imagenes_cargadas = {}


def cargar_imagen(ruta):
    """
    Carga las imágenes y las gaurda en la variable global
    """
    if ruta not in imagenes_cargadas:
        imagenes_cargadas[ruta] = cv2.imread(ruta) if ruta else None

    return imagenes_cargadas[ruta]


def imagen_default(dificultad):
    """
    Carga solamente las imágenes llamadas "default". Son las que usamos para comenzar 
    la partida de dicha dificultad
    """
    for extension in (".jpg", ".jpeg", ".png"):
        ruta = os.path.join("media", dificultad, "default" + extension)
        if os.path.exists(ruta):
            return ruta

    return None


def escala(frame):
    """
    Escala
    """
    return frame.shape[1] / 640.0


def texto(frame, txt, x, y, color=(255, 255, 255), tam=0.7, grosor=2, centrado=False):
    """
    Texto con contorno negro para que se lea sobre cualquier fondo, porque blanco vimos que 
    no es tan buena opción si tu casa es blanca
    """
    tam *= escala(frame)
    grosor = max(1, int(round(grosor * escala(frame))))
    if centrado:
        (ancho_texto, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, tam, grosor)
        x = x - ancho_texto / 2

    cv2.putText(frame, txt, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, tam, (0, 0, 0), grosor + 3, cv2.LINE_AA)
    cv2.putText(frame, txt, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, tam, color, grosor, cv2.LINE_AA)


def oscurecer(frame, cantidad):
    """
    Oscurece el fondo. Usado para la explicación y las notas
    """
    frame[:] = (frame * (1 - cantidad)).astype(np.uint8)


def pegar(frame, imagen, x, y, ancho, borde=None):
    """
    Pega la imagen reescalada a 'ancho' con su esquina de arriba a la izquierda en (x, y).
    """
    if imagen is None:
        return x, y, 0, 0

    alto_imagen = int(imagen.shape[0] * ancho / imagen.shape[1])
    alto, ancho_visible = min(alto_imagen, frame.shape[0] - y), min(ancho, frame.shape[1] - x) # se recorta si se sale
    if alto <= 0 or ancho_visible <= 0:
        return x, y, 0, 0

    frame[y:y + alto, x:x + ancho_visible] = cv2.resize(imagen, (ancho, alto_imagen))[:alto, :ancho_visible]
    if borde is not None:
        cv2.rectangle(frame, (x, y), (x + ancho_visible - 1, y + alto - 1), borde, max(2, int(3 * escala(frame))))

    return x, y, ancho_visible, alto


def barra(frame, x, y, ancho, alto, progreso, color=(0, 220, 0)): # color verde 
    """
    Barra de progreso para la selección de dificultad
    """
    cv2.rectangle(frame, (x, y), (x + ancho, y + alto), (0, 0, 0), -1)
    cv2.rectangle(frame, (x, y), (x + int(ancho * max(0.0, min(1.0, progreso))), y + alto), color, -1)
    cv2.rectangle(frame, (x, y), (x + ancho, y + alto), (255, 255, 255), 1)


def color_nota(nota):
    """
    Da un color según la puntuación de tu postura
    """
    if nota >= UMBRAL_APROBADO:
        return (0, 220, 0) # Verde 

    return (0, 220, 255) if nota >= UMBRAL_APROBADO - 20 else (0, 0, 255) # Devuelve amarillo si cumple un umbral de semi-acierto, rojo si lo has hecho de pena


def marco_rojo(frame, nadie):
    """
    Marco rojo alrededor de la imagen cuando no se ve al jugador o solo se le ve en parte.
    """
    alto, ancho = frame.shape[0], frame.shape[1]
    grosor = max(8, int(12 * escala(frame)))
    cv2.rectangle(frame, (0, 0), (ancho - 1, alto - 1), (0, 0, 255), grosor)

    if nadie: # en el centro, total no hay nadie a quien tapar.
        texto(frame, "No se te ve: colocate delante de la camara", ancho / 2, alto / 2, (0, 0, 255), 0.7, centrado=True)
    else:
        texto(frame, "No se te ve entero: alejate o centrate", ancho / 2, alto * 0.22, (0, 0, 255), 0.6, centrado=True)


def pantalla_menu(frame, estado):
    """
    Función principal para mostrar el menú completo.
    """
    
    alto, ancho = frame.shape[0], frame.shape[1]
    margen = int(10 * escala(frame))
    ancho_miniatura = int(ancho * TAMANO_MINIATURA)
    menu = estado["menu"]

    for dificultad, esquina in ESQUINAS_MENU.items(): # Imprime cada una de las imágenes y barras de progreso por dificultad.
        imagen = cargar_imagen(imagen_default(dificultad))
        alto_miniatura = int(imagen.shape[0] * ancho_miniatura / imagen.shape[1]) if imagen is not None else ancho_miniatura

        # Se determina dónde se colocará la imagen
        x = margen if esquina.endswith("izq") else ancho - ancho_miniatura - margen
        y = margen + int(22 * escala(frame)) if esquina.startswith("arriba") else alto - alto_miniatura - margen - int(22 * escala(frame))

        elegida = dificultad == menu["candidata"]
        _, _, w, h = pegar(frame, imagen, x, y, ancho_miniatura, borde=(0, 220, 0) if elegida else (255, 255, 255))

        puntuacion = menu["puntuaciones"].get(dificultad)
        etiqueta = dificultad.upper() + ("" if puntuacion is None else f"  {puntuacion:.0f}")
        texto(frame, etiqueta, x, y - int(6 * escala(frame)), (0, 220, 0) if elegida else (255, 255, 255), 0.55)

        if elegida: # Colorea el progreso de la selección
            barra(frame, x, y + h + int(4 * escala(frame)), w, int(10 * escala(frame)), menu["progreso"])

    texto(frame, "Q: salir", margen, alto - margen - int(8 * escala(frame)), (255, 255, 255), 0.6)
    texto(frame, "Haz la postura de una esquina para elegir", ancho / 2, alto - margen - int(40 * escala(frame)), (255, 255, 255), 0.55, centrado=True)


def pantalla_explicacion(frame, estado):
    """
    Función para mostrar la imagen que le tocará al usuario simular.
    """
    alto, ancho = frame.shape[0], frame.shape[1]
    oscurecer(frame, 0.75)
    pose = estado["pose"]
    texto(frame, f"Postura {estado['indice'] + 1} de {estado['total']}: {pose['nombre']}", ancho / 2, int(40 * escala(frame)), (255, 255, 255), 0.8, centrado=True)

    imagen = cargar_imagen(pose["imagen"])
    if imagen is not None: # lo más grande posible sin pasarse del 70 % del alto ni del 80 % del ancho
        alto_max, ancho_max = int(alto * 0.7), int(ancho * 0.8)
        ancho_imagen = min(ancho_max, int(imagen.shape[1] * alto_max / imagen.shape[0]))
        alto_imagen = int(imagen.shape[0] * ancho_imagen / imagen.shape[1])
        pegar(frame, imagen, (ancho - ancho_imagen) // 2, int(60 * escala(frame)) + (alto_max - alto_imagen) // 2, ancho_imagen, borde=(255, 255, 255))

    texto(frame, f"Fijate bien... {estado['restante']:.0f}", ancho / 2, alto - int(20 * escala(frame)), (255, 255, 255), 0.7, centrado=True)


def pantalla_ejecucion(frame, estado):
    """
    Función para mostrar la pantalla durante la ejecución de una pose. 
    """
    alto, ancho = frame.shape[0], frame.shape[1]
    margen = int(10 * escala(frame))
    ancho_miniatura = int(ancho * TAMANO_MINIATURA)

    pegar(frame, cargar_imagen(estado["pose"]["imagen"]), ancho - ancho_miniatura - margen, margen, ancho_miniatura, borde=(255, 255, 255))
    texto(frame, f"{estado['indice'] + 1}/{estado['total']} {estado['pose']['nombre']}", margen, int(28 * escala(frame)), (255, 255, 255), 0.6)

    # Contador
    color = (0, 0, 255) if estado["restante"] <= 5 else (255, 255, 255)
    texto(frame, f"Quedan {estado['restante']:.0f} s", margen, int(58 * escala(frame)), color, 0.7)

    comparacion = estado["comparacion"]
    if comparacion and comparacion["puntuacion"] is not None: # Si hay comparación, pone la nota con su color
        texto(frame, f"{comparacion['puntuacion']:.0f}", margen, int(95 * escala(frame)), color_nota(comparacion["puntuacion"]), 1.0)

    ancho_barra = int(ancho * 0.5)
    texto(frame, "Manten la postura", ancho / 2, alto - int(42 * escala(frame)), (255, 255, 255), 0.55, centrado=True)
    barra(frame, (ancho - ancho_barra) // 2, alto - int(34 * escala(frame)), ancho_barra, int(14 * escala(frame)), estado["mantenido"] / VENTANA_PUNTUACION)


def pantalla_nota(frame, estado):
    """
    Muestra la nota que has obtenido.
    """
    alto, ancho = frame.shape[0], frame.shape[1]
    oscurecer(frame, 0.6)
    texto(frame, estado["pose"]["nombre"], ancho / 2, alto / 2 - int(40 * escala(frame)), (255, 255, 255), 0.9, centrado=True)
    texto(frame, f"{estado['nota']:.0f}", ancho / 2, alto / 2 + int(30 * escala(frame)), color_nota(estado["nota"]), 2.2, 4, centrado=True)


def pantalla_resultados(frame, estado):
    """
    Muestra el resultado final de la partida, con los reslutados anteriores y la media final.
    """
    alto, ancho = frame.shape[0], frame.shape[1]
    oscurecer(frame, 0.8)
    e = escala(frame) # con 'escala' falla
    texto(frame, f"Resultados - {str(estado['dificultad']).upper()}", ancho / 2, int(45 * e), (255, 255, 255), 0.9, centrado=True)

    y = int(95 * e)
    for resultado in estado["resultados"]: # Muestra en orden los resultados
        texto(frame, resultado["nombre"], ancho * 0.18, y, (255, 255, 255), 0.65)
        texto(frame, f"{resultado['nota']:.0f}", ancho * 0.75, y, color_nota(resultado["nota"]), 0.75)
        y += int(34 * e)

    # Muestra la media
    cv2.line(frame, (int(ancho * 0.15), y - int(18 * e)), (int(ancho * 0.85), y - int(18 * e)), (255, 255, 255), 1)
    media = estado.get("media", 0.0) 
    texto(frame, "Media", ancho * 0.18, y + int(12 * e), (255, 255, 255), 0.8)
    texto(frame, f"{media:.0f}", ancho * 0.75, y + int(12 * e), color_nota(media), 1.0)
    texto(frame, "ENTER: volver al menu     Q: salir", ancho / 2, alto - int(20 * e), (150, 150, 150), 0.55, centrado=True) # Color gris 


# Este diccionario es para ejecutar una serie de funciones acorde a la fase (que sería la clave)
# Como todos reciben frame y estado, se puede hacer esta maravilla. Gracias Python TQM
PANTALLAS = {
    "menu": pantalla_menu,
    "explicacion": pantalla_explicacion,
    "ejecucion": pantalla_ejecucion,
    "nota_pose": pantalla_nota,
    "resultados": pantalla_resultados
}


def dibujar(frame, estado, result):
    """
    Dibuja el esqueleto y la pantalla de la fase en la que esté el juego.
    Sorprendentemente limpio la verdad.
    """
    if result.pose_landmarks and estado["fase"] in ("menu", "ejecucion"):
        dibujar_esqueleto(frame, result.pose_landmarks[0], None, estado["comparacion"])

    PANTALLAS[estado["fase"]](frame, estado)

    if estado["fase"] in ("nota_pose", "resultados"):
        return

    if not estado["detectado"]:
        marco_rojo(frame, nadie=True)
    
    elif not estado.get("visible", True):
        marco_rojo(frame, nadie=False)