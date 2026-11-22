"""
Aquí se maneja la lógica del juego. Como lo pone en la memoria, es una especie de reto de posturas para flexibilidad. Antes preferíamos solo de yoga, pero
hemos optado por poner poses generales también o incluso otras posiciones famosas como las de Jojo Adventures.

Lógica del juego (sin dibujo: eso está en ui.py). main.py llama a Juego.actualizar() una vez por fotograma.

Fases:
    MENU         -> se elige la dificultad haciendo la pose "default" de una esquina (selector_dificultad.py)
    EXPLICACION  -> T_EXPLICACION s mostrando la foto de la pose
    EJECUCION    -> hasta T_LIMITE[dificultad] s (30/45/60) puntuando desde el principio. Termina antes en
                    cuanto se mantiene la pose >= UMBRAL_APROBADO durante VENTANA_PUNTUACION s seguidos
    NOTA_POSE    -> T_RESULTADO_POSE s con la nota de esa pose, y a la siguiente
    RESULTADOS   -> nota de cada pose y media final

Nota de una pose = media del MEJOR tramo de VENTANA_PUNTUACION s seguidos dentro de la ejecución:
    - una caída o el rato de colocarse solo estropean los tramos que los contienen, no la nota;
    - hacerlo muy bien un momento no basta: 3 s a 95 y 2 s a 30 dan (3*95 + 2*30) / 5 = 69, suspenso.
Fotogramas sin persona detectada cuentan como 0; los "no evaluables" (persona vista pero sin suficientes
ángulos medibles) no cuentan ni a favor ni en contra.
"""
import os

from poses import comparar_pose, cargar_postura
from selector_dificultad import SelectorDificultad, cargar_poses_seleccion
from utils import (T_EXPLICACION, T_LIMITE, VENTANA_PUNTUACION, GRACIA_MANTENER,
                   COBERTURA_VENTANA, T_RESULTADO_POSE, UMBRAL_APROBADO, NOMBRES_POSES)

MENU, EXPLICACION, EJECUCION, NOTA_POSE, RESULTADOS = (
    "menu", "explicacion", "ejecucion", "nota_pose", "resultados")
EXTENSIONES = (".jpg", ".jpeg", ".png")


def mejor_ventana(muestras, ventana=VENTANA_PUNTUACION, cobertura=COBERTURA_VENTANA):
    """
    muestras: [(t, puntuación o None)] en orden de tiempo. Devuelve la media del mejor tramo de `ventana`
    segundos (solo cuentan las puntuaciones no None, y el tramo necesita al menos `cobertura` de ellas).
    Si la grabación dura menos que la ventana, se usa entera. Sin ningún tramo válido devuelve 0.
    """
    if not muestras:
        return 0.0
    t_fin = muestras[-1][0]
    if t_fin - muestras[0][0] < ventana:
        tramos = [[v for _, v in muestras]]                               # todo es un único tramo
    else:
        tramos = [[v for t, v in muestras[i:] if t < t0 + ventana]
                  for i, (t0, _) in enumerate(muestras) if t0 + ventana <= t_fin + 1e-6]
    mejor = None
    for tramo in tramos:
        validos = [v for v in tramo if v is not None]
        if tramo and validos and len(validos) / len(tramo) >= cobertura:
            media = sum(validos) / len(validos)
            mejor = media if mejor is None else max(mejor, media)
    return 0.0 if mejor is None else mejor


def _imagen_de(dificultad, nombre, carpeta_media):
    for ext in EXTENSIONES:
        ruta = os.path.join(carpeta_media, dificultad, nombre + ext)
        if os.path.exists(ruta):
            return ruta
    return None


def cargar_partida(dificultad, carpeta_json="jason", carpeta_media="media"):
    """Poses de la partida: jason/<dificultad>/p*.json en orden (la default es la del menú y no se juega)."""
    carpeta = os.path.join(carpeta_json, dificultad)
    nombres = sorted(f[:-5] for f in os.listdir(carpeta) if f.endswith(".json") and f != "default.json")
    partida = []
    for n, nombre in enumerate(nombres, 1):
        clave = f"{dificultad}/{nombre}"
        partida.append({
            "clave": clave,
            "nombre": NOMBRES_POSES.get(clave, f"Postura {n}"),
            "postura": cargar_postura(os.path.join(carpeta, nombre + ".json")),
            "imagen": _imagen_de(dificultad, nombre, carpeta_media),
        })
    return partida


class Juego:
    def __init__(self, carpeta_json="jason", carpeta_media="media"):
        self.carpeta_json, self.carpeta_media = carpeta_json, carpeta_media
        self.selector = SelectorDificultad(cargar_poses_seleccion(carpeta_json))
        self.ir_al_menu()

    # --- cambios de fase -------------------------------------------------------------------------
    def ir_al_menu(self):
        self.fase, self.t_fase = MENU, None
        self.selector.reiniciar()
        self.dificultad, self.partida, self.indice, self.resultados = None, [], 0, []

    def _empezar(self, dificultad, ahora):
        self.dificultad = dificultad
        self.partida = cargar_partida(dificultad, self.carpeta_json, self.carpeta_media)
        self.indice, self.resultados = 0, []
        self._fase(EXPLICACION if self.partida else RESULTADOS, ahora)

    def _duracion(self, fase):
        return {EXPLICACION: T_EXPLICACION, EJECUCION: T_LIMITE.get(self.dificultad, 30),
                NOTA_POSE: T_RESULTADO_POSE}[fase]

    def _fase(self, fase, ahora):
        self.fase, self.t_fase = fase, ahora
        if fase == EJECUCION:
            self.muestras = []                   # (segundos desde el inicio de la ejecución, puntuación)
            self.mantener_inicio = self.mantener_ultimo_ok = None

    def _terminar_pose(self, ahora):
        pose = self.partida[self.indice]
        self.resultados.append({"nombre": pose["nombre"], "clave": pose["clave"],
                                "nota": mejor_ventana(self.muestras)})
        self._fase(NOTA_POSE, ahora)

    # --- un fotograma ----------------------------------------------------------------------------
    def actualizar(self, result, ancho, alto, ahora):
        """Avanza el juego con el esqueleto de este fotograma y devuelve el estado que ui.py dibuja."""
        detectado = bool(result.pose_landmarks)
        estado = {"fase": self.fase, "detectado": detectado, "dificultad": self.dificultad,
                  "comparacion": None, "restante": None, "mantenido": 0.0, "pose": None,
                  "indice": self.indice, "total": len(self.partida), "resultados": self.resultados}

        if self.fase == MENU:
            estado["menu"] = self.selector.actualizar(result, ancho, alto, ahora)
            if estado["menu"]["elegida"]:
                self._empezar(estado["menu"]["elegida"], ahora)
            return estado

        if self.fase == RESULTADOS:
            notas = [r["nota"] for r in self.resultados]
            estado["media"] = sum(notas) / len(notas) if notas else 0.0
            return estado

        pose = self.partida[self.indice]
        estado["pose"] = pose
        transcurrido = ahora - self.t_fase
        estado["restante"] = max(0.0, self._duracion(self.fase) - transcurrido)

        if self.fase == EJECUCION and detectado:
            estado["comparacion"] = comparar_pose(result, pose["postura"], ancho, alto)

        if self.fase == EXPLICACION and transcurrido >= T_EXPLICACION:
            self._fase(EJECUCION, ahora)

        elif self.fase == EJECUCION:
            punt = 0.0 if not detectado else estado["comparacion"]["puntuacion"]   # None = no evaluable
            self.muestras.append((transcurrido, punt))
            # Cuánto lleva mantenida la pose por encima del umbral (perdonando parpadeos cortos)
            if punt is not None and punt >= UMBRAL_APROBADO:
                if self.mantener_inicio is None:
                    self.mantener_inicio = ahora
                self.mantener_ultimo_ok = ahora
            elif self.mantener_inicio is not None and ahora - self.mantener_ultimo_ok > GRACIA_MANTENER:
                self.mantener_inicio = self.mantener_ultimo_ok = None
            if self.mantener_inicio is not None:
                estado["mantenido"] = self.mantener_ultimo_ok - self.mantener_inicio
            if estado["mantenido"] >= VENTANA_PUNTUACION or transcurrido >= self._duracion(EJECUCION):
                self._terminar_pose(ahora)

        elif self.fase == NOTA_POSE:
            estado["nota"] = self.resultados[-1]["nota"]
            if transcurrido >= T_RESULTADO_POSE:
                self.indice += 1
                self._fase(EXPLICACION if self.indice < len(self.partida) else RESULTADOS, ahora)

        return estado