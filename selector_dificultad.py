"""
Menú de dificultad por postura: el jugador elige la dificultad HACIENDO la pose "default" de esa carpeta
(jason/<dificultad>/default.json) y manteniéndola.

Para que no se elija una dificultad sin querer (la de fácil es una T, muy fácil de hacer de pasada):
    - hay que superar UMBRAL_SELECCION (más exigente que el del juego, UMBRAL_APROBADO),
    - y mantenerlo SEGUNDOS_SELECCION seguidos.
Si la puntuación cae por debajo un instante (ruido del modelo), se perdona hasta GRACIA_SELECCION segundos;
si cae más, o se pasa a otra pose, el contador vuelve a 0.

Uso (cada frame):
    selector = SelectorDificultad(cargar_poses_seleccion())
    estado = selector.actualizar(result, ancho, alto, time.monotonic())
    if estado["elegida"]:
        dificultad = estado["elegida"]          # "facil", "medio" o "dificil"
"""
import os

from poses import comparar_pose, cargar_postura

DIFICULTADES = ("facil", "medio", "dificil")
UMBRAL_SELECCION = 75.0      # puntuación mínima (0-100) para que cuente como "haciendo" la pose del menú
SEGUNDOS_SELECCION = 3.0     # cuánto hay que mantenerla para elegir
GRACIA_SELECCION = 0.5       # segundos que se perdonan por debajo del umbral sin reiniciar el contador


def cargar_poses_seleccion(carpeta="jason"):
    """{dificultad: postura} con las default.json que existan."""
    poses = {}
    for dificultad in DIFICULTADES:
        ruta = os.path.join(carpeta, dificultad, "default.json")
        if os.path.exists(ruta):
            poses[dificultad] = cargar_postura(ruta)
    return poses


class SelectorDificultad:
    def __init__(self, poses, umbral=UMBRAL_SELECCION, segundos=SEGUNDOS_SELECCION, gracia=GRACIA_SELECCION):
        self.poses = poses
        self.umbral, self.segundos, self.gracia = umbral, segundos, gracia
        self.reiniciar()

    def reiniciar(self):
        self.candidata = None        # dificultad cuya pose se está manteniendo
        self.inicio = None           # instante en que empezó a mantenerla
        self.ultimo_ok = None        # último instante en que estuvo por encima del umbral
        self.elegida = None

    def actualizar(self, result, ancho, alto, ahora):
        """
        Evalúa el frame contra las poses del menú y avanza el contador. `ahora` en segundos (time.monotonic()).
        Devuelve {"puntuaciones": {dif: punt o None}, "candidata", "progreso" (0-1), "elegida" (o None)}.
        """
        puntuaciones = {}
        for dificultad, pose in self.poses.items():
            puntuaciones[dificultad] = comparar_pose(result, pose, ancho, alto)["puntuacion"] if result.pose_landmarks else None

        if self.elegida is None:
            validas = {d: p for d, p in puntuaciones.items() if p is not None and p >= self.umbral}
            mejor = max(validas, key=validas.get) if validas else None

            if mejor is not None and mejor == self.candidata:
                self.ultimo_ok = ahora                                   # sigue con la misma: acumula
            elif mejor is not None:
                self.candidata, self.inicio, self.ultimo_ok = mejor, ahora, ahora   # empieza otra: desde 0
            elif self.candidata is not None and ahora - self.ultimo_ok > self.gracia:
                self.candidata, self.inicio, self.ultimo_ok = None, None, None      # la ha soltado

            if self.candidata is not None and self.ultimo_ok - self.inicio >= self.segundos:
                self.elegida = self.candidata

        progreso = 0.0
        if self.elegida is not None:
            progreso = 1.0
        elif self.candidata is not None:
            progreso = min(1.0, (self.ultimo_ok - self.inicio) / self.segundos)
        return {"puntuaciones": puntuaciones, "candidata": self.candidata, "progreso": progreso,
                "elegida": self.elegida}