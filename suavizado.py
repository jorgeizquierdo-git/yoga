"""
Reducción de ruido del esqueleto en tiempo real.

MediaPipe predice cada fotograma casi por separado, así que los puntos tiemblan aunque el jugador esté quieto,
y de vez en cuando uno "salta" a otro sitio durante un instante. Este módulo filtra los 33 puntos ANTES de
calcular ángulos, así que se estabilizan a la vez el dibujo, los ángulos, los colores y la puntuación.

Dos mecanismos, por punto:

1. TEMBLORES -> filtro One Euro (Casiez et al., 2012). Es un paso bajo que adapta su corte a la velocidad:
   - quieto  -> corte bajo  -> suaviza mucho (en yoga casi siempre se está quieto),
   - moviéndose -> corte alto -> casi no añade retraso.
   Además, cuanto MENOS visible es un punto (p. ej. la pierna tapada en una pose de perfil), más se suaviza:
   esos puntos son los que más tiemblan.

2. SALTOS y CAÍDAS DE VISIBILIDAD -> mantener el último valor bueno.
   Si un punto se desplaza en un solo fotograma más de SALTO_MAXIMO (medido en "troncos", la distancia
   hombros-caderas, para que no dependa de lo lejos que esté el jugador), o su visibilidad cae por debajo de
   MIN_VISIBILIDAD, se ignora ese valor y se mantiene el último bueno. Si la situación dura más de
   MANTENER_MAX_S, se acepta (el movimiento era real, o el punto de verdad ya no se ve).

Uso (en el bucle, justo después de detectar):
    filtro = FiltroEsqueleto()
    ...
    result = landmarker.detect_for_video(mp_image, timestamp_ms)
    result = filtro.aplicar(result, time.monotonic())
El resultado tiene la misma forma que el de MediaPipe (pose_landmarks[0][i].x/.y/.z/.visibility), así que
el resto del código no cambia.
"""
import math
from types import SimpleNamespace

from utils import (MIN_VISIBILIDAD, FILTRO_ACTIVO, FILTRO_MIN_CUTOFF, FILTRO_BETA, FILTRO_D_CUTOFF, SALTO_MAXIMO, MANTENER_MAX_S, IDX)


def _alfa(corte_hz, dt):
    """Coeficiente del suavizado exponencial equivalente a un paso bajo de frecuencia de corte `corte_hz`."""
    tau = 1.0 / (2 * math.pi * corte_hz)
    return 1.0 / (1.0 + tau / dt)


class OneEuro:
    """Filtro One Euro para un valor escalar."""

    def __init__(self, min_cutoff, beta, d_cutoff):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x = None       # último valor filtrado
        self.dx = 0.0       # última derivada filtrada

    def reiniciar(self, x):
        self.x, self.dx = x, 0.0

    def __call__(self, x, dt, factor_corte=1.0):
        if self.x is None or dt <= 0:
            self.reiniciar(x)
            return x
        dx = (x - self.x) / dt
        self.dx += _alfa(self.d_cutoff, dt) * (dx - self.dx)
        corte = (self.min_cutoff + self.beta * abs(self.dx)) * factor_corte
        self.x += _alfa(corte, dt) * (x - self.x)
        return self.x


class _Punto:
    def __init__(self):
        self.fx = OneEuro(FILTRO_MIN_CUTOFF, FILTRO_BETA, FILTRO_D_CUTOFF)
        self.fy = OneEuro(FILTRO_MIN_CUTOFF, FILTRO_BETA, FILTRO_D_CUTOFF)
        self.ultimo = None          # (x, y, z, visibilidad) de salida del último fotograma bueno
        self.t_bueno = None         # instante del último fotograma bueno
        self.sospechoso_desde = None


class FiltroEsqueleto:
    def __init__(self, activo=FILTRO_ACTIVO):
        self.activo = activo
        self.puntos = None
        self.t_prev = None
        self.t_ultima_persona = None

    def reiniciar(self):
        self.puntos, self.t_prev = None, None

    def _tronco(self):
        """Distancia media hombro-cadera de la última salida (normalizada); None si aún no hay."""
        try:
            lados = []
            for lado in ("izq", "der"):
                h = self.puntos[IDX[f"hombro_{lado}"]].ultimo
                c = self.puntos[IDX[f"cadera_{lado}"]].ultimo
                lados.append(math.hypot(h[0] - c[0], h[1] - c[1]))
            return max(sum(lados) / 2, 1e-3)
        except (TypeError, KeyError):
            return None

    def aplicar(self, result, ahora):
        if not self.activo:
            return result
        if not result.pose_landmarks:
            # Si la persona desaparece un rato, se olvida el estado para no "arrastrar" la pose anterior
            if self.t_ultima_persona is not None and ahora - self.t_ultima_persona > MANTENER_MAX_S:
                self.reiniciar()
            return result
        self.t_ultima_persona = ahora

        crudos = result.pose_landmarks[0]
        if self.puntos is None or len(self.puntos) != len(crudos):
            self.puntos = [_Punto() for _ in crudos]
        dt = (ahora - self.t_prev) if self.t_prev is not None else 0.0
        self.t_prev = ahora
        tronco = self._tronco()

        salida = []
        for p, est in zip(crudos, self.puntos):
            sospechoso = False
            if est.ultimo is not None:
                if p.visibility < MIN_VISIBILIDAD:
                    sospechoso = True                                   # el modelo dejó de verlo
                elif tronco is not None:
                    salto = math.hypot(p.x - est.ultimo[0], p.y - est.ultimo[1]) / tronco
                    sospechoso = salto > SALTO_MAXIMO                    # se ha "teletransportado"

            if sospechoso:
                if est.sospechoso_desde is None:
                    est.sospechoso_desde = ahora
                if ahora - est.sospechoso_desde <= MANTENER_MAX_S:
                    salida.append(SimpleNamespace(x=est.ultimo[0], y=est.ultimo[1], z=est.ultimo[2], visibility=est.ultimo[3]))
                    continue
                # Dura demasiado: es real. Se acepta y, si era un salto, se reinicia el filtro en la nueva posición
                if p.visibility >= MIN_VISIBILIDAD:
                    est.fx.reiniciar(p.x); est.fy.reiniciar(p.y)
            est.sospechoso_desde = None

            # Menos visibilidad -> corte más bajo -> más suavizado (entre el 30 % y el 100 % del corte)
            factor = 0.3 + 0.7 * min(max(p.visibility, 0.0), 1.0)
            x = est.fx(p.x, dt, factor)
            y = est.fy(p.y, dt, factor)
            est.ultimo = (x, y, p.z, p.visibility)
            est.t_bueno = ahora
            salida.append(SimpleNamespace(x=x, y=y, z=p.z, visibility=p.visibility))

        return SimpleNamespace(pose_landmarks=[salida], pose_world_landmarks=getattr(result, "pose_world_landmarks", None))