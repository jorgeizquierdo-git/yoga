"""
Sonidos del juego: carga los .wav de CARPETA_SONIDOS (soniditos/) y los reproduce con sounddevice.

- Se reproducen en segundo plano (sd.play vuelve al instante), así que no congelan el vídeo.
- Si sounddevice o PortAudio no están disponibles, o falla el dispositivo de audio, el juego sigue en silencio
  (se avisa una vez por consola). En Linux, PortAudio se instala con: sudo apt install libportaudio2
- Si suena un sonido mientras otro aún no ha terminado, el nuevo corta al anterior (son todos muy cortos).

Uso:
    sonidos = Sonidos()
    sonidos.reproducir("conseguida")              # nombre del .wav sin extensión
    sonidos.reproducir_eventos(estado["eventos"]) # de varios a la vez, solo suena el más importante
"""
import os
import wave

import numpy as np

from utils import SONIDOS_ACTIVOS, CARPETA_SONIDOS, VOLUMEN_SONIDOS

try:
    import sounddevice as sd
except (ImportError, OSError):      # OSError: sounddevice instalado pero falta PortAudio
    sd = None


# Si en un mismo fotograma tocan varios, suena solo el primero de esta lista (uno cortaría al otro)
PRIORIDAD = ["fin_partida", "conseguida", "tiempo_agotado", "dificultad_elegida", "nueva_postura",
             "mantener_5", "mantener_4", "mantener_3", "mantener_2", "mantener_1", "no_se_te_ve", "tic"]


def _cargar_wav(ruta):
    """Lee un .wav PCM de 16 bits con la librería estándar -> (muestras float32 en [-1, 1], frecuencia)."""
    with wave.open(ruta, "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError("solo se admiten .wav de 16 bits")
        datos = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
        if w.getnchannels() > 1:
            datos = datos.reshape(-1, w.getnchannels())
        return datos, w.getframerate()


class Sonidos:
    def __init__(self, carpeta=CARPETA_SONIDOS, activos=SONIDOS_ACTIVOS):
        self.sonidos = {}
        self.activos = activos and sd is not None
        if activos and sd is None:
            print("[sonidos] sounddevice/PortAudio no disponible: el juego irá en silencio "
                  "(pip install sounddevice; en Linux: sudo apt install libportaudio2)")
        if not os.path.isdir(carpeta):
            print(f"[sonidos] No existe la carpeta {carpeta}/: el juego irá en silencio")
            self.activos = False
            return
        for archivo in sorted(os.listdir(carpeta)):
            if archivo.lower().endswith(".wav"):
                try:
                    self.sonidos[archivo[:-4]] = _cargar_wav(os.path.join(carpeta, archivo))
                except (wave.Error, ValueError, EOFError) as e:
                    print(f"[sonidos] No se pudo cargar {archivo}: {e}")

    def reproducir(self, nombre):
        if not self.activos or nombre not in self.sonidos:
            return
        datos, frecuencia = self.sonidos[nombre]
        try:
            sd.play(datos * VOLUMEN_SONIDOS, frecuencia)      # no bloquea
        except Exception as e:                                # p. ej. sin dispositivo de salida
            print(f"[sonidos] Error al reproducir ({e}): se desactiva el sonido")
            self.activos = False

    def reproducir_eventos(self, eventos):
        """De los eventos de este fotograma, reproduce solo el de más prioridad."""
        if eventos:
            self.reproducir(min(eventos, key=lambda e: PRIORIDAD.index(e) if e in PRIORIDAD else len(PRIORIDAD)))