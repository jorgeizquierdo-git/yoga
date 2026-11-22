import os
import wave
import numpy as np
from utils import SONIDOS_ACTIVOS, CARPETA_SONIDOS
import sounddevice as sd

# Si en el mismo frame tocan varios sonidos solo suena el primero de esta lista, porque uno cortaría al otro
# Es, en esencia, una cola de prioridad.
PRIORIDAD = [
    "fin_partida", "conseguida", "tiempo_agotado", "dificultad_elegida", "nueva_postura",
    "mantener_5", "mantener_4", "mantener_3", "mantener_2", "mantener_1", "no_se_te_ve", "tic"
]


def cargar_wav(ruta):
    """
    Lee un archivo .wav y devuelve las muestras entre -1 y 1 y su frecuencia
    """
    with wave.open(ruta, "rb") as w:
        datos = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
        if w.getnchannels() > 1:
            datos = datos.reshape(-1, w.getnchannels())

        return datos, w.getframerate()


class Sonidos:
    """
    Esta clase está diseñada para reproducir el sonido correspondiente. 
    Durante el trabajo, nos percatamos que, entre variables globales y funciones, al
    final nos convenía una clase
    """
    def __init__(self):
        """
        Inicilaización. Carga todos los sonidos
        """
        self.sonidos = {}
        self.activos = SONIDOS_ACTIVOS

        for archivo in sorted(os.listdir(CARPETA_SONIDOS)):
            if archivo.endswith(".wav"):
                self.sonidos[archivo[:-4]] = cargar_wav(os.path.join(CARPETA_SONIDOS, archivo))
                # archivo[:-4] quita del nombre el .wav

    def reproducir(self, nombre):
        """
        No bloquea el juego. Si falla el audio se desactiva y el juego sigue en silencio
        """
        if nombre not in self.sonidos: # Y este sonido pallo??!??!?!?!?! FUERA
            return

        datos, frecuencia = self.sonidos[nombre]
        try:
            sd.play(datos, frecuencia)
        except Exception as e: # Sospechamos que puede fallar y reventar el código. Por eso hacemos este handler
            print(f"Error al reproducir sonidos ({e}), se desactiva el sonido")
            self.activos = False

    def reproducir_eventos(self, eventos):
        """
        De los sonidos que tocan en este frame solo reproduce el más importante
        """
        if eventos:
            self.reproducir(min(eventos, key=lambda e: PRIORIDAD.index(e) if e in PRIORIDAD else len(PRIORIDAD)))
            # Gracias miji por tu sabiduria con los lambda