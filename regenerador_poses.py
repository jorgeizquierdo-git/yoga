"""
Genera (o regenera) los .json de las posturas a partir de las imágenes de media/<dificultad>/.

    media/<dificultad>/<nombre>.jpg|.jpeg|.png   ->   jason/<dificultad>/<nombre>.json

- Si el .json NO existe, se crea con todos los ángulos de utils.ANGULOS como relevantes, tolerancia TOL_DEFECTO
  y los pesos de PESOS_GENERALES + CONFIG_POSES (abajo).
- Si el .json YA existe, se vuelven a medir solo los "objetivo" y se CONSERVA lo que hayáis ajustado a mano:
  qué ángulos son relevantes, "tol", "peso", "nombre" y "mantener_s". (Para volver a aplicar los pesos de
  CONFIG_POSES, borrad ese .json y regeneradlo.)
- Si el .json tiene "no_regenerar": true en "postura" (p. ej. una referencia hecha con vuestras capturas),
  no se toca.

Uso (desde la carpeta del repo, con el modelo descargado):
    python regenerador_poses.py                          -> todas las imágenes
    python regenerador_poses.py dificil/p2               -> solo esa (se pueden poner varias)
    python regenerador_poses.py dificil/p2 --nuevo       -> ignora el .json que hubiera y lo crea desde cero
                                                           (pesos y ángulos relevantes de CONFIG_POSES)

Si algo sale mal, los .json se recuperan con git (git checkout -- jason/).
"""
import json
import sys
from pathlib import Path

from registrador_poses import registrar_pose_desde_imagen
from utils import MODEL_PATH, ANGULOS

CARPETA_JSON = Path("jason")
CARPETA_MEDIA = Path("media")
EXTENSIONES = (".jpg", ".jpeg", ".png")
TOL_DEFECTO = 10
MANTENER_DEFECTO = 10

# Pesos que se aplican a TODAS las posturas nuevas (por tipo de ángulo; vale para los dos lados).
# tobillo y cabeza_tronco usan segmentos muy cortos (pie, cuello): unos pocos píxeles de error en la foto
# son 15-20 grados, así que cuentan la mitad para que no arruinen una pose bien hecha.
PESOS_GENERALES = {"tobillo": 0.5, "cabeza_tronco": 0.5}

# Pesos específicos de cada postura (mandan sobre PESOS_GENERALES). Clave: "<dificultad>/<nombre del archivo>".
# Se puede usar el tipo ("cadera": ambos lados) o la clave exacta ("rodilla_der").
# Idea: lo que DEFINE la postura pesa 2-3; lo que se hace sin querer (p. ej. piernas rectas) pesa poco.
CONFIG_POSES = {
    # --- fácil ---
    "facil/default": {"pesos": {"hombro": 2, "codo": 1.5}},                             # T: brazos en cruz
    "facil/p1":      {"pesos": {"rodilla": 3, "cadera": 2, "hombro": 2}},               # árbol
    "facil/p2":      {"pesos": {"rodilla": 2, "hombro": 2, "cadera": 1.5}},             # guerrero II
    "facil/p3":      {"pesos": {"rodilla": 2.5, "cadera": 2.5, "hombro": 1.5}},         # silla
    "facil/p4":      {"pesos": {"cadera": 3, "hombro": 2, "rodilla": 1.5}},             # perro boca abajo
    # --- medio ---
    # niño, de perfil: el lado izquierdo de la foto está tapado y su esqueleto no es de fiar
    "medio/default": {"pesos": {"cadera": 2.5, "rodilla": 2, "hombro": 2},
                      "relevantes": ["rodilla_der", "cadera_der", "codo_der", "hombro_der", "tobillo_der",
                                     "cabeza_tronco_der"]},
    "medio/p1":      {"pesos": {"cadera": 3, "rodilla": 2, "hombro": 1.5}},             # puente
    "medio/p2":      {"pesos": {"cadera": 3, "hombro": 2, "codo": 1.5, "rodilla": 0.5}},  # cobra: piernas rectas valen poco
    "medio/p3":      {"pesos": {"cadera": 2, "hombro": 2, "rodilla": 1.5}},             # triángulo
    "medio/p4":      {"pesos": {"rodilla": 2, "cadera": 2, "hombro": 1}},               # lagarto / zancada baja
    # --- difícil ---
    "dificil/default": {"pesos": {"cadera": 3, "hombro": 2, "rodilla": 1.5}},           # media luna
    "dificil/p1":      {"pesos": {"rodilla": 2, "cadera": 2, "codo": 1.5, "hombro": 1.5}},  # zancada cogiendo el pie
    "dificil/p2":      {"pesos": {"rodilla": 2.5, "cadera": 2}},                        # salto
    # invertida: brazos y cabeza salen colapsados en la foto -> cuentan poco
    "dificil/p3":      {"pesos": {"cadera": 3, "rodilla": 2, "hombro": 0.5, "codo": 0.5, "cabeza_tronco": 0.25}},
    "dificil/p4":      {"pesos": {"cadera": 2.5, "hombro": 2.5, "rodilla": 1.5}},       # wild thing / plancha lateral
}


def _imagenes():
    return sorted(p for p in CARPETA_MEDIA.glob("*/*") if p.suffix.lower() in EXTENSIONES)


def _cargar(ruta_json):
    if not ruta_json.exists():
        return None
    with ruta_json.open(encoding="utf-8") as f:
        return json.load(f)["postura"]


def regenerar(imagen, nuevo=False):
    clave_pose = f"{imagen.parent.name}/{imagen.stem}"
    ruta_json = CARPETA_JSON / imagen.parent.name / f"{imagen.stem}.json"
    antigua = None if nuevo else _cargar(ruta_json)
    config = CONFIG_POSES.get(clave_pose, {})

    if antigua and antigua.get("no_regenerar"):
        return f"SALTADA     {ruta_json} (no_regenerar)"

    if antigua:
        # Relevante = tiene "tol" (aunque la otra vez no se pudiera medir: con otro modelo puede que sí se mida).
        # Los que se quitan a mano llevan "tol": null y no vuelven.
        relevantes = [c for c, d in antigua["angulos"].items()
                      if d.get("objetivo") is not None or d.get("tol") is not None]
        if not relevantes:            # todos salieron "no evaluable" la otra vez: se reintenta con todos
            relevantes = list(ANGULOS["angulos"].keys())
    else:
        relevantes = config.get("relevantes", list(ANGULOS["angulos"].keys()))
    pesos = {**PESOS_GENERALES, **config.get("pesos", {})}

    try:
        contenido = registrar_pose_desde_imagen(
            ruta_imagen=imagen,
            ruta_modelo=MODEL_PATH,
            ruta_salida_json=ruta_json,
            nombre_pose=(antigua or {}).get("nombre", config.get("nombre", clave_pose)),
            articulaciones_relevantes=relevantes,
            tolerancia=config.get("tol", TOL_DEFECTO),
            mantener_s=(antigua or {}).get("mantener_s", config.get("mantener_s", MANTENER_DEFECTO)),
            pesos=pesos,
        )
    except ValueError as e:
        return f"ERROR       {imagen}: {e}" + (" (el .json anterior se deja como estaba)" if antigua else "")

    angulos = contenido["postura"]["angulos"]
    # Si ya existía, se respetan la tolerancia y el peso que tuviera cada ángulo
    if antigua:
        for clave in relevantes:
            viejo = antigua["angulos"].get(clave, {})
            for campo in ("tol", "peso"):
                if viejo.get(campo) is not None and clave in angulos:
                    angulos[clave][campo] = viejo[campo]
        with ruta_json.open("w", encoding="utf-8") as f:
            json.dump(contenido, f, indent=2, ensure_ascii=False)

    # Resumen para revisar a ojo
    lineas = []
    for clave in relevantes:
        d = angulos[clave]
        valor = "NO EVALUABLE" if d["objetivo"] is None else f"{d['objetivo']:6.1f}"
        anterior = ""
        if antigua and antigua["angulos"].get(clave, {}).get("objetivo") is not None:
            anterior = f"  (antes {antigua['angulos'][clave]['objetivo']})"
        lineas.append(f"    {clave:<20} {valor}  peso={d['peso']}{anterior}")
    sin_medir = [c for c in relevantes if angulos[c]["objetivo"] is None]
    aviso = f"  <-- {len(sin_medir)} sin medir, revisadla" if sin_medir else ""
    estado = "ACTUALIZADA" if antigua else "NUEVA      "
    return f"{estado} {ruta_json} <- {imagen}{aviso}\n" + "\n".join(lineas)


def _buscar(clave_pose):
    """"dificil/p2" (con o sin extensión) -> ruta de su imagen en media/, o None si no existe."""
    base = CARPETA_MEDIA / Path(clave_pose).with_suffix("")
    for ext in EXTENSIONES:
        if base.with_suffix(ext).exists():
            return base.with_suffix(ext)
    return None


def main(poses=None, nuevo=False):
    """
    poses: lista de "<dificultad>/<nombre>" (p. ej. ["dificil/p2"]). Si es None o vacía, se hacen todas.
    nuevo: True para no tener en cuenta el .json existente (se crea desde cero con CONFIG_POSES).
    """
    if poses:
        imagenes = []
        for clave in poses:
            imagen = _buscar(clave)
            if imagen is None:
                print(f"No existe ninguna imagen {CARPETA_MEDIA}/{clave}{{{','.join(EXTENSIONES)}}}")
            else:
                imagenes.append(imagen)
    else:
        imagenes = _imagenes()
    if not imagenes:
        print(f"No hay imágenes que procesar en {CARPETA_MEDIA}/<dificultad>/")
        return
    for imagen in imagenes:
        print(regenerar(imagen, nuevo=nuevo))
    sin_config = [f"{i.parent.name}/{i.stem}" for i in imagenes if f"{i.parent.name}/{i.stem}" not in CONFIG_POSES]
    if sin_config:
        print("\nSin pesos específicos en CONFIG_POSES (solo PESOS_GENERALES):", ", ".join(sin_config))


if __name__ == "__main__":
    argumentos = sys.argv[1:]
    main([a for a in argumentos if not a.startswith("--")], nuevo="--nuevo" in argumentos)