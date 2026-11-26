"""
Suavizado del esqueleto en tiempo real.

EL PROBLEMA
MediaPipe analiza cada fotograma prácticamente por separado, sin "recordar" dónde estaban los puntos en el anterior.
Esto provoca dos fallos distintos que vimos al probar el juego:

    1. Temblor: aunque el jugador esté completamente quieto, los puntos vibran unos píxeles de un fotograma a otro.
       Como los ángulos se calculan con esos puntos, también oscilan, y un ángulo que está justo en el límite de la
       tolerancia pasa de verde a rojo continuamente. Con el esqueleto "crudo" la evaluación era casi imposible.
    2. Saltos: de vez en cuando un punto (sobre todo uno tapado, como la pierna de atrás en una pose de perfil)
       aparece durante uno o dos fotogramas en un sitio que no tiene nada que ver, y luego vuelve.

DÓNDE SE APLICA
El filtro se aplica a los 33 puntos justo después de detectarlos y ANTES de calcular los ángulos (main.py y
tracker_poses.py). Así se estabiliza todo a la vez: el dibujo del esqueleto, los ángulos, los colores y la puntuación.

LA SOLUCIÓN: un mecanismo para cada problema, aplicado a cada punto por separado

    1. Contra el temblor -> filtro One Euro (Casiez, Roussel y Vogel, 2012).
       Es un filtro paso bajo (deja pasar los cambios lentos y elimina los rápidos, que son el temblor) cuya
       frecuencia de corte cambia según lo rápido que se mueve el punto:
           - Punto quieto      -> corte bajo  -> suaviza mucho. En yoga se está quieto casi todo el rato.
           - Punto moviéndose  -> corte alto  -> suaviza poco, así el esqueleto no se queda "atrasado".
       Un filtro con un corte fijo obliga a elegir entre mucho temblor o mucho retraso; este evita el dilema.
       Además, cuanto MENOS visible es un punto según MediaPipe, más se suaviza, porque esos son los que más tiemblan.

    2. Contra los saltos -> mantener el último valor bueno.
       Un valor nuevo se considera "sospechoso" si:
           - su visibilidad baja de MIN_VISIBILIDAD (el modelo dice que no lo ve bien), o
           - el punto se ha desplazado más de SALTO_MAXIMO en un solo fotograma. Esta distancia se mide en "troncos"
             (distancia hombro-cadera) y no en píxeles, para que el umbral valga igual estés cerca o lejos de la cámara.
       Mientras un punto sea sospechoso, se ignora el valor nuevo y se repite el último bueno. Si la situación dura
       más de MANTENER_MAX_S segundos, se asume que es real (el jugador se ha movido de verdad o el punto de verdad
       ha dejado de verse) y se acepta.

PARÁMETROS (están en utils.py para poder ajustarlos sin tocar este archivo)
    FILTRO_ACTIVO       True/False. Con False el esqueleto sale "crudo"; sirve para comparar con y sin filtro.
    FILTRO_MIN_CUTOFF   Corte con el punto quieto (Hz). Más bajo = suaviza más, pero tarda más en llegar a la pose.
    FILTRO_BETA         Cuánto sube el corte con la velocidad. Más alto = menos retraso al moverse, más temblor.
    FILTRO_D_CUTOFF     Corte del filtro que suaviza la velocidad (Hz). No hace falta tocarlo.
    SALTO_MAXIMO        Desplazamiento máximo en UN fotograma, en troncos, antes de considerarlo un salto.
    MANTENER_MAX_S      Segundos que se ignora un salto o una caída de visibilidad antes de aceptarlos.

CÓMO SE USA (en el bucle principal, justo después de detectar)
    filtro = FiltroEsqueleto()
    ...
    result = landmarker.detect_for_video(mp_image, timestamp_ms)
    result = filtro.aplicar(result, time.monotonic())

El resultado tiene la misma forma que el de MediaPipe (result.pose_landmarks[0][i].x / .y / .z / .visibility),
así que el resto del código no ha tenido que cambiar para usar el filtro.
"""
import math
from types import SimpleNamespace

from utils import (MIN_VISIBILIDAD, FILTRO_ACTIVO, FILTRO_MIN_CUTOFF, FILTRO_BETA, FILTRO_D_CUTOFF, SALTO_MAXIMO, MANTENER_MAX_S, IDX)


def _alfa(corte_hz, dt):
    """
    Convierte una frecuencia de corte (en Hz) en el coeficiente alfa del suavizado exponencial.

    El suavizado exponencial es: nuevo_filtrado = filtrado + alfa * (valor_nuevo - filtrado)
        - alfa = 1. No se filtra nada (se copia el valor nuevo).
        - alfa ~= 0. El valor apenas se mueve (mucho suavizado, mucho retraso).

    La fórmula es la de un filtro paso bajo de primer orden discretizado:
        tau = 1 / (2 * pi * corte)       (constante de tiempo del filtro)
        alfa = 1 / (1 + tau / dt)
    Se usa dt (el tiempo real entre fotogramas) para que el filtro se comporte igual sin depender de los FPS

    Ejemplo de camara a 30FPS (dt = 0.033 s):
        corte = 0.3 Hz -> alfa ~ 0.06 Caso punto quieto y casi invisible -> suavizado altísimo
        corte = 1 Hz -> alfa ~ 0.17 Punto quieto y visible -> suavizado normal
        corte = 3.5 Hz -> alfa ~ 0.42 Punto moviéndose -> suavizado bajo
    """
    tau = 1.0 / (2 * math.pi * corte_hz)
    return 1.0 / (1.0 + tau / dt)


class OneEuro:
    """
    Filtro One Euro para UN valor (por ejemplo, la coordenada x de la rodilla izquierda).
    Cada punto del esqueleto usa dos: uno para x y otro para y.
    """

    def __init__(self, min_cutoff, beta, d_cutoff):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x = None # último valor filtrado
        self.dx = 0.0 # última velocidad filtrada (el grado de cambio)

    def reiniciar(self, x):
        """
        Coloca el filtro directamente en x sin suavizar y con velocidad 0. Sirve para la inicialización del esquelto o cuando 
        un "salto" resultó ser un movimiento real. En caso contrario de no usar esta función, el filtro arrastraría poco a poco
        desde su antigua posición hasta la actual.
        """
        self.x, self.dx = x, 0.0

    def __call__(self, x, dt, factor_corte=1.0):
        """
        Filtra el nuevo x y devuelve su valor suavizado.
            - dt son los segundos desde el fotograma anterior.
            - factor_corte multiplica la propia frecuencia de corte. Lo usamos para suavizar más los puntos con poca visibilidad
        """
        if self.x is None or dt <= 0: # primer valor (o tiempo no válido). No se suaviza
            self.reiniciar(x)
            return x

        # Velocidad del punto que se mide como el cambio de dicho punto entre el fotograma anterior y el actual (en segundos).
        # Las coordenadas están normalizadas a la imagen, por lo que la velocidad va en "imágenes por segundo" (una velocidad de 1 significa que, 
        # a ese ritmo, el punto cruzaría la imagen entera en un segundo).
        dx = (x - self.x) / dt

        # La velocidad calculada con puntos que tiemblan es más ruidosa que los puntos en sí mismo, por lo que se suavizan
        # igualmente, con un corte fijo
        self.dx += _alfa(self.d_cutoff, dt) * (dx - self.dx)

        # El corte es una función que crece con la velocidad: Cuando se está quieto, corte = min_cutoff (mucho suavizado). 
        # Moviendose tiene un corte mayor (pco retraso). 
        # Si se moviera a 0.5 imagenes / s el corte sería (1 + 5 * 0.5) = 3.5 Hz.
        corte = (self.min_cutoff + self.beta * abs(self.dx)) * factor_corte

        # Suavizado exponencial del valor con el alfa correspondiente a este corte. x_nuevo = filtro + alpha * (x_nuevo - filtrado)
        self.x += _alfa(corte, dt) * (x - self.x)
        return self.x


class _Punto:
    """
    Estado que se guarda de un punto del esqueleto entre fotogramas.
    """
    def __init__(self):
        self.fx = OneEuro(FILTRO_MIN_CUTOFF, FILTRO_BETA, FILTRO_D_CUTOFF) # Filtro de la coordenada x
        self.fy = OneEuro(FILTRO_MIN_CUTOFF, FILTRO_BETA, FILTRO_D_CUTOFF) # Filtro de la coordenada y
        self.ultimo = None # (x, y, z, visibilidad) del último fotograma bueno. Es lo que se repite ante un salto
        self.t_bueno = None # Instante del último fotograma bueno
        self.sospechoso_desde = None # Instante en que el punto empezó a ser sospechoso


class FiltroEsqueleto:
    """
    Filtro del esqueleto completo: guarda un objeto de la clase _Punto por cada uno de los 33 puntos y los filtra en cada fotograma.
    """
    def __init__(self, activo=FILTRO_ACTIVO):
        self.activo = activo
        self.puntos = None # Lista de puntos. Se crea al detectar a la primera persona
        self.t_prev = None # Instante del fotograma anterior para calcular dt
        self.t_ultima_persona = None # Último instante en que se detectó a alguien

    def reiniciar(self):
        """
        Olvida todo el estado.
        """
        self.puntos, self.t_prev = None, None

    def _tronco(self):
        """
        Longitud media del tronco (distancia hombro-cadera) dentro del último fotograma bueno. Sirve de regla para medir los saltos.
        Si el jugador está lejos el tronco es pequeño y vice versa si está cerca. Dividir el tronco permite que el umbral del salto
        no dependa de la distancia a la cámara. Devuelve None en caso de que no haya suficientes datos.
        """
        try:
            lados = []
            for lado in ("izq", "der"):
                h = self.puntos[IDX[f"hombro_{lado}"]].ultimo
                c = self.puntos[IDX[f"cadera_{lado}"]].ultimo
                lados.append(math.hypot(h[0] - c[0], h[1] - c[1])) # distancia euclídea en 2D. Si fuera un array más grande usaríamos np.linalg.norm
            return max(sum(lados) / 2, 1e-3) # 1e-3 para no dividir nunca entre cero
        except (TypeError, KeyError): # TypeError: algún 'ultimo' todavía es None
            return None

    def aplicar(self, result, ahora):
        """
        Devuelve un resultado con la misma forma que el de MediaPipe pero con los puntos suavizados.
        Filtra el resultado de Mediapipe de un fotograma concreto. "Result" es el esqueleto resultante de dicho fotograma, mientras
        que "ahora" es el instante actual (con time.monotonic())
        """
        if not self.activo: # Si está desactivado el filtro se devuelve tal cual
            return result

        if not result.pose_landmarks: # No se detecta a nadie en este fotograma
            # Si la persona desaparece más de MANTENER_MAX_S, se olvida su esqueleto. Así, cuando regrese
            # no se mezcla con la pose anterior ni se suaviza desde una posición inutil.
            if self.t_ultima_persona is not None and ahora - self.t_ultima_persona > MANTENER_MAX_S:
                self.reiniciar()
            return result
        self.t_ultima_persona = ahora

        crudos = result.pose_landmarks[0]
        if self.puntos is None or len(self.puntos) != len(crudos): # Se crea los objetos de _Punto si es la primera vez
            self.puntos = [_Punto() for _ in crudos]
        dt = (ahora - self.t_prev) if self.t_prev is not None else 0.0 # dt = 0 hace que los filtros se inicialicen
        self.t_prev = ahora
        tronco = self._tronco() # se calcula una vez por fotograma, con los puntos del fotograma anterior

        salida = []
        for p, est in zip(crudos, self.puntos): # itera los puntos crudos y los estados guardados
            # MECANISMO 2: detectar si el valor nuevo es sospechoso (solo si ya hay un valor bueno anterior)
            sospechoso = False
            if est.ultimo is not None:
                if p.visibility < MIN_VISIBILIDAD:
                    sospechoso = True # el modelo dejó de ver el punto
                
                elif tronco is not None:
                    salto = math.hypot(p.x - est.ultimo[0], p.y - est.ultimo[1]) / tronco # desplazamiento en troncos
                    sospechoso = salto > SALTO_MAXIMO # se ha "teletransportado"

            if sospechoso: # Si resulta sospechoso
                if est.sospechoso_desde is None: # Acaba de empezar a ser sospechoso, por lo que se guarda cuándo
                    est.sospechoso_desde = ahora
                
                if ahora - est.sospechoso_desde <= MANTENER_MAX_S: # Todavía dentro del margen: se ignora el valor nuevo y se repite el último bueno
                    salida.append(SimpleNamespace(x=est.ultimo[0], y=est.ultimo[1], z=est.ultimo[2], visibility=est.ultimo[3]))
                    continue
                
                # Dura demasiado. Por ende, es real. Se acepta y, si era un salto (el punto se ve bien), se reinician los filtros
                # en la nueva posición para que el punto vaya directamente allí en vez de deslizarse poco a poco.
                if p.visibility >= MIN_VISIBILIDAD:
                    est.fx.reiniciar(p.x); est.fy.reiniciar(p.y)
            est.sospechoso_desde = None # el valor se acepta y deja de ser sospechoso

            # MECANISMO 1: suavizado One Euro.
            # Menos visibilidad -> corte más bajo -> más suavizado. El factor va del 30 % (visibilidad 0)
            # al 100 % (visibilidad 1) del corte normal.
            factor = 0.3 + 0.7 * min(max(p.visibility, 0.0), 1.0) # minmax por si la visibilidad se sale del intervalo [0, 1]
            x = est.fx(p.x, dt, factor)
            y = est.fy(p.y, dt, factor)
            # La z no se filtra ya que los ángulos se calculan en 2D (x, y), así que no la usamos para evaluar.
            est.ultimo = (x, y, p.z, p.visibility)
            est.t_bueno = ahora
            salida.append(SimpleNamespace(x=x, y=y, z=p.z, visibility=p.visibility))

        # Se devuelve con la misma estructura que MediaPipe (SimpleNamespace imita sus objetos), así el resto del código
        # puede usarlo sin saber si viene filtrado o no.
        return SimpleNamespace(pose_landmarks=[salida], pose_world_landmarks=getattr(result, "pose_world_landmarks", None))