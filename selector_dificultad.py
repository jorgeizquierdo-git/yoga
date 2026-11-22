import os
from poses import comparar_pose, cargar_postura


DIFICULTADES = ("facil", "medio", "dificil")
UMBRAL_SELECCION = 75.0 # Nuevo umbral de puntaje para seleccionar. Así, evitamos que haga posturas "por suerte" o sin querer
SEGUNDOS_SELECCION = 3.0
GRACIA_SELECCION = 0.5 # segundos por debajo del umbral que se perdonan sin reiniciar el contador


def cargar_poses_seleccion(carpeta="jason"):
    """
    Carga las posturas default de cada dificultad. Sinceramente, no hay mucha
    magia aquí. Solamente que son las 11 de la noche y quiero dormir
    """
    poses = {}
    for dificultad in DIFICULTADES:
        ruta = os.path.join(carpeta, dificultad, "default.json")
        
        if os.path.exists(ruta):
            poses[dificultad] = cargar_postura(ruta)

    return poses


class SelectorDificultad:
    """
    Basandonos en soniditos.py usamos tambien una clase para las dificultades.
    """
    def __init__(self, poses):
        """
        Aquí ya recibe las posiciones enteras
        """
        self.poses = poses
        self.reiniciar()

    def reiniciar(self):
        self.candidata = None # dificultad cuya postura se está manteniendo
        self.inicio = None
        self.ultimo_ok = None # último momento en que estuvo por encima del umbral
        self.elegida = None # dificultad definitivamente elegida

    def actualizar(self, result, ancho, alto, ahora):
        """
        Compara el frame con las posturas del menú y avanza el contador. 2x1 básicamente
        """
        puntuaciones = {}
        for dificultad, pose in self.poses.items():
            puntuaciones[dificultad] = comparar_pose(result, pose, ancho, alto)["puntuacion"] if result.pose_landmarks else None

        if self.elegida is None:
            validas = {dif: punto for dif, punto in puntuaciones.items() if punto is not None and punto >= UMBRAL_SELECCION}
            mejor = max(validas, key=validas.get) if validas else None

            if mejor is not None and mejor == self.candidata: # sigue con la misma candidata
                self.ultimo_ok = ahora

            elif mejor is not None: # Hay nuevo candidato. 
                self.candidata, self.inicio, self.ultimo_ok = mejor, ahora, ahora

            elif self.candidata is not None and ahora - self.ultimo_ok > GRACIA_SELECCION: # descarta la que tenias como candidata si superas el umbral de tiempo
                self.candidata, self.inicio, self.ultimo_ok = None, None, None

            if self.candidata is not None and self.ultimo_ok - self.inicio >= SEGUNDOS_SELECCION: # si la candidata se mantuvo lo suficiente, se selecciona definitivamente
                self.elegida = self.candidata

        progreso = 0.0
        if self.elegida is not None:
            progreso = 1.0
        
        elif self.candidata is not None:
            progreso = min(1.0, (self.ultimo_ok - self.inicio) / SEGUNDOS_SELECCION)

        return {
            "puntuaciones": puntuaciones,
            "candidata": self.candidata,
            "progreso": progreso, # Progreso para seleccionarla. Por eso, si elegida is not None, proceso = 1
            "elegida": self.elegida
        }