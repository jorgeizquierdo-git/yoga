"""
Regenera todos los .json de jason/<dificultad>/ a partir de su foto en media/<dificultad>/ (mismo nombre, .jpg).

Hace falta tras pasar los ángulos de 3D (world) a 2D (píxeles): los "objetivo" antiguos se midieron con
otro método y ya no son comparables con lo que mide el juego.

Se conserva de cada .json antiguo:
    - qué ángulos eran relevantes (los que tenían "objetivo" distinto de null),
    - la tolerancia ("tol") de cada uno,
    - "nombre" y "mantener_s".
Solo cambian los valores de "objetivo" (y se añade "esqueleto" con los 33 puntos).

Uso (desde la carpeta del repo, con el modelo descargado):
    python regenerar_poses.py

Los .json se sobrescriben: si algo sale mal, se recuperan con git (git checkout -- jason/).
"""
import json
from pathlib import Path

from registrador_poses import registrar_pose_desde_imagen
from utils import MODEL_PATH

CARPETA_JSON = Path("jason")
CARPETA_MEDIA = Path("media")
EXTENSIONES = (".jpg", ".jpeg", ".png")


def _buscar_imagen(ruta_json):
    """media/<dificultad>/<nombre>.jpg|.jpeg|.png correspondiente a jason/<dificultad>/<nombre>.json."""
    base = CARPETA_MEDIA / ruta_json.parent.name / ruta_json.stem
    for ext in EXTENSIONES:
        if base.with_suffix(ext).exists():
            return base.with_suffix(ext)
    return None


def regenerar(ruta_json):
    imagen = _buscar_imagen(ruta_json)
    if imagen is None:
        return f"SIN FOTO   {ruta_json} (se deja como estaba)"

    with ruta_json.open(encoding="utf-8") as f:
        antigua = json.load(f)["postura"]

    relevantes = [clave for clave, d in antigua["angulos"].items() if d.get("objetivo") is not None]
    if not relevantes:
        return f"SIN ANGULOS RELEVANTES {ruta_json} (se deja como estaba)"

    try:
        contenido = registrar_pose_desde_imagen(
            ruta_imagen=imagen,
            ruta_modelo=MODEL_PATH,
            ruta_salida_json=ruta_json,
            nombre_pose=antigua.get("nombre", ruta_json.stem),
            articulaciones_relevantes=relevantes,
            mantener_s=antigua.get("mantener_s", 10),
        )
    except ValueError as e:
        return f"ERROR      {ruta_json}: {e} (se deja como estaba)"

    # registrar_pose_desde_imagen usa una tolerancia común: restauramos la de cada ángulo
    angulos = contenido["postura"]["angulos"]
    for clave in relevantes:
        if clave in angulos:
            angulos[clave]["tol"] = antigua["angulos"][clave].get("tol", angulos[clave]["tol"])
    with ruta_json.open("w", encoding="utf-8") as f:
        json.dump(contenido, f, indent=2, ensure_ascii=False)

    # Cambios de objetivo, para revisar a ojo
    cambios = []
    for clave in relevantes:
        viejo = antigua["angulos"][clave]["objetivo"]
        nuevo = angulos.get(clave, {}).get("objetivo")
        nuevo_txt = "no evaluable" if nuevo is None else f"{nuevo:.1f}"
        cambios.append(f"    {clave:<20} {viejo:>6} -> {nuevo_txt}")
    return f"OK         {ruta_json} <- {imagen}\n" + "\n".join(cambios)


def main():
    rutas = sorted(CARPETA_JSON.glob("*/*.json"))
    rutas = [r for r in rutas if r.parent.name != "capturas"]   # las capturas no tienen foto
    if not rutas:
        print(f"No hay .json en {CARPETA_JSON}/<dificultad>/")
        return
    for ruta in rutas:
        print(regenerar(ruta))


if __name__ == "__main__":
    main()