import math
import os
from poses import comparar_pose, cargar_postura
from selector_dificultad import SelectorDificultad, cargar_poses_seleccion
from utils import (
    T_EXPLICACION, T_LIMITE, T_RESULTADO_POSE, VENTANA_PUNTUACION, GRACIA_MANTENER, COBERTURA_VENTANA, UMBRAL_APROBADO,
    NOMBRES_POSES, AVISO_AUSENCIA_CADA, AUSENCIA_MINIMA, TICS_FINALES
)

def mejor_ventana(muestras):
    """
    Media del mejor tramo de VENTANA_PUNTUACION segundos seguidos. Los None (no evaluable) no cuentan y un tramo
    necesita al menos COBERTURA_VENTANA de valores válidos. Si se ha durado menos que la ventana, cuenta todo
    """
    if not muestras:
        return 0.0

    t_fin = muestras[-1][0] # Ultima muestra
    if t_fin - muestras[0][0] < VENTANA_PUNTUACION: # Si tarda menos que toda la ventana temporal, lo cuenta todo
        tramos = [[valor for _, valor in muestras]]
    else: # Por cada frame, crea un tramo
        tramos = [[valor for t, valor in muestras[i:] if t < t0 + VENTANA_PUNTUACION] 
                  for i, (t0, _) in enumerate(muestras) if t0 + VENTANA_PUNTUACION <= t_fin + 1e-6]

    medias = []
    for tramo in tramos: # Por cada tramo, toma sus valores 
        validos = [valor for valor in tramo if valor is not None]
        if validos and len(validos) / len(tramo) >= COBERTURA_VENTANA: # Si supera la cobertura de analisis de frames, se gudarda
            medias.append(sum(validos) / len(validos))

    return max(medias, default=0.0) # Toma el mayor de las medidas 


def buscar_imagen(dificultad, nombre):
    """
    Ya no quiero explicar esta funcion. Me niego
    """
    for extension in (".jpg", ".jpeg", ".png"):
        ruta = os.path.join("media", dificultad, nombre + extension)
        if os.path.exists(ruta):
            return ruta

    return None


def cargar_partida(dificultad):
    """
    Carga las posturas de la dificultad, excepto la default.
    """
    carpeta = os.path.join("jason", dificultad)
    nombres = sorted(f[:-5] for f in os.listdir(carpeta) if f.endswith(".json") and f != "default.json")
    
    return [{
        "clave": f"{dificultad}/{nombre}",
        "nombre": NOMBRES_POSES.get(f"{dificultad}/{nombre}", f"Postura {n}"),
        "postura": cargar_postura(os.path.join(carpeta, nombre + ".json")),
        "imagen": buscar_imagen(dificultad, nombre),
    } for n, nombre in enumerate(nombres, 1)]


class Juego:
    """
    Aquí si que es obvio crear una clase del juego que mantenga el estado del mismo y 
    aplice la lógica con sus métodos.
    """
    def __init__(self):
        """
        Su selector es un objeto de SelectorDificultad. Tendrá una lista de eventos que suceden.
        Comienza en el menú
        """
        self.selector = SelectorDificultad(cargar_poses_seleccion())
        self.eventos = []
        self.ir_al_menu()

    def ir_al_menu(self):
        """
        Reinicia casi todos sus atributos
        """
        self.fase, self.t_fase = "menu", None
        self.selector.reiniciar()
        self.dificultad = None
        self.partida = []
        self.indice = 0
        self.resultados = []

    def cambiar_fase(self, fase, ahora):
        """
        Cambio de los atributos según el estado al que pasarán
        """
        self.fase, self.t_fase = fase, ahora
        
        if fase == "explicacion":
            self.eventos.append("nueva_postura")
        
        elif fase == "resultados":
            self.eventos.append("fin_partida")
        
        elif fase == "ejecucion":
            self.muestras = [] # (segundos desde el inicio, puntuación)
            self.mantener_inicio = self.mantener_ultimo_ok = None
            self.segundos_sonados = 0 # cuántos "tic" han sonado ya
            self.ultimo_tic = self.ausente_desde = self.ultimo_aviso_ausencia = None

    def duracion(self, fase):
        """
        Devuelve el tiempo correspondiente a la fase actual. Creo que no hacía falta realmente
        esta función
        """
        return {
            "explicacion": T_EXPLICACION, 
            "ejecucion": T_LIMITE.get(self.dificultad, 30), 
            "nota_pose": T_RESULTADO_POSE
        }[fase]

    def actualizar(self, result, ancho, alto, ahora):
        """
        Avanza el juego un frame y devuelve el estado que dibuja ui.py
        """
        self.eventos = []
        detectado = bool(result.pose_landmarks)
        estado = { # Estado default.
            "fase": self.fase, 
            "detectado": detectado, 
            "dificultad": self.dificultad, 
            "comparacion": None,
            "restante": None, 
            "mantenido": 0.0, 
            "pose": None, 
            "indice": self.indice, 
            "total": len(self.partida),
            "resultados": self.resultados, 
            "eventos": self.eventos
        }

        if self.fase == "menu":
            estado["menu"] = self.selector.actualizar(result, ancho, alto, ahora)
            
            if estado["menu"]["elegida"]:
                self.eventos.append("dificultad_elegida")
                self.dificultad = estado["menu"]["elegida"]
                self.partida, self.indice, self.resultados = cargar_partida(self.dificultad), 0, []
                self.cambiar_fase("explicacion" if self.partida else "resultados", ahora)
            
            return estado

        if self.fase == "resultados":
            notas = [r["nota"] for r in self.resultados]
            estado["media"] = sum(notas) / len(notas) if notas else 0.0
            
            return estado

        pose = self.partida[self.indice]
        transcurrido = ahora - self.t_fase
        estado["pose"] = pose
        estado["restante"] = max(0.0, self.duracion(self.fase) - transcurrido)

        if self.fase == "ejecucion" and detectado:
            estado["comparacion"] = comparar_pose(result, pose["postura"], ancho, alto)

        # Determina si se le ve lo suficiente para ver al usuario. Se tiene en cuenta si aún no se puede comparar con nada
        estado["visible"] = detectado and not (estado["comparacion"] and estado["comparacion"]["estado"] == "no_evaluable")

        # Cadena de ifs para pasar de una fase a otra
        if self.fase == "explicacion" and transcurrido >= T_EXPLICACION:
            self.cambiar_fase("ejecucion", ahora)

        elif self.fase == "ejecucion":
            self.ejecutar(estado, detectado, transcurrido, ahora)

        elif self.fase == "nota_pose":
            estado["nota"] = self.resultados[-1]["nota"]
            
            if transcurrido >= T_RESULTADO_POSE:
                self.indice += 1
                self.cambiar_fase("explicacion" if self.indice < len(self.partida) else "resultados", ahora)

        return estado

    def ejecutar(self, estado, detectado, transcurrido, ahora):
        """
        Un frame de la ejecución: guarda la puntuación, cuenta lo que llevas mantenido y decide qué sonidos tocan
        """
        puntuacion = estado["comparacion"]["puntuacion"] if detectado else 0.0 # None = no evaluable, no cuenta
        self.muestras.append((transcurrido, puntuacion))

        if puntuacion is not None and puntuacion >= UMBRAL_APROBADO: # A+
            self.mantener_inicio = self.mantener_inicio or ahora
            self.mantener_ultimo_ok = ahora
        
        elif self.mantener_inicio is not None and ahora - self.mantener_ultimo_ok > GRACIA_MANTENER: # El manco ha perdido su postura xdxdxdxd
            self.mantener_inicio = self.mantener_ultimo_ok = None
            self.segundos_sonados = 0

        if self.mantener_inicio is not None:
            estado["mantenido"] = self.mantener_ultimo_ok - self.mantener_inicio
            segundo = int(estado["mantenido"]) + 1 # un pitido cada segundo, cada vez más agudo
        
            if self.segundos_sonados < segundo <= VENTANA_PUNTUACION:
                self.eventos.append(f"mantener_{min(segundo, 5)}")
                self.segundos_sonados = segundo

        restante = math.ceil(estado["restante"])
        if 0 < restante <= TICS_FINALES and restante != self.ultimo_tic: # Pitidos para que el usuario regrese al juego
            self.eventos.append("tic")
            self.ultimo_tic = restante

        if estado["visible"]:
            self.ausente_desde = None
        
        else: # se avisa a partir de AUSENCIA_MINIMA y luego cada AVISO_AUSENCIA_CADA
            self.ausente_desde = self.ausente_desde or ahora
        
            if ahora - self.ausente_desde >= AUSENCIA_MINIMA and (self.ultimo_aviso_ausencia is None or ahora - self.ultimo_aviso_ausencia >= AVISO_AUSENCIA_CADA):
                self.eventos.append("no_se_te_ve")
                self.ultimo_aviso_ausencia = ahora

        conseguida = estado["mantenido"] >= VENTANA_PUNTUACION
        
        if conseguida or transcurrido >= self.duracion("ejecucion"):
            self.eventos.append("conseguida" if conseguida else "tiempo_agotado")
            self.resultados.append({
                "nombre": self.partida[self.indice]["nombre"], 
                "clave": self.partida[self.indice]["clave"], 
                "nota": mejor_ventana(self.muestras)
                })
            self.cambiar_fase("nota_pose", ahora)