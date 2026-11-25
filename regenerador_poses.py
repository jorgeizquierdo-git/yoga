import json
import sys
from pathlib import Path
from registrador_poses import registrar_pose_desde_imagen
from utils import MODEL_PATH, ANGULOS

EXTENSIONES = (".jpg", ".jpeg", ".png") # Extensiones permitidas

PESOS_GENERALES = {"tobillo": 0.5, "cabeza_tronco": 0.5} # Pesos para angulos irregulares / imprecisos
CONFIG_POSES = {
    "facil/default": {"pesos": {"hombro": 2, "codo": 1.5}},
    "facil/p1": {"pesos": {"rodilla": 3, "cadera": 2, "hombro": 2}},
    "facil/p2": {"pesos": {"rodilla": 2, "hombro": 2, "cadera": 1.5}},
    "facil/p3": {"pesos": {"rodilla": 2.5, "cadera": 2.5, "hombro": 1.5}},
    "facil/p4": {"pesos": {"cadera": 3, "hombro": 2, "rodilla": 1.5}},
    "medio/default": {"pesos": {"cadera": 2.5, "rodilla": 2, "hombro": 2}},
    "medio/p1": {"pesos": {"cadera": 3, "rodilla": 2, "hombro": 1.5}},
    "medio/p2": {"pesos": {"cadera": 3, "hombro": 2, "codo": 1.5, "rodilla": 0.5}},
    "medio/p3": {"pesos": {"cadera": 2, "hombro": 2, "rodilla": 1.5}},
    "medio/p4": {"pesos": {"rodilla": 2, "cadera": 2, "hombro": 1}},
    "dificil/default": {"pesos": {"cadera": 3, "hombro": 2, "rodilla": 1.5}},
    "dificil/p1": {"pesos": {"rodilla": 2, "cadera": 2, "codo": 1.5, "hombro": 1.5}},
    "dificil/p2": {"pesos": {"rodilla": 2.5, "cadera": 2}},
    "dificil/p3": {"pesos": {"cadera": 3, "rodilla": 2, "hombro": 0.5, "codo": 0.5, "cabeza_tronco": 0.25}},
    "dificil/p4": {"pesos": {"cadera": 2.5, "hombro": 2.5, "rodilla": 1.5}},
}

def buscar_imagen(clave_pose):
    """
    Busca la imagen de la posición.
    """
    base = Path("media") / clave_pose

    for extension in EXTENSIONES:
        if base.with_suffix(extension).exists():
            return base.with_suffix(extension)
    
    return None


def cargar_postura_antigua(ruta_json):
    """
    Carga la postura antigua.
    """
    if not ruta_json.exists():
        return None
    
    with ruta_json.open(encoding="utf-8") as f:
        return json.load(f)["postura"]


def regenerar(imagen, nuevo=False):
    """
    Genera el JSON de una imagen y devuelve un resumen de los ángulos para revisarlo a ojo
    """
    clave_pose = f"{imagen.parent.name}/{imagen.stem}"
    ruta_json = Path("jason") / imagen.parent.name / f"{imagen.stem}.json"
    antigua = None if nuevo else cargar_postura_antigua(ruta_json)
    config = CONFIG_POSES.get(clave_pose, {})

    if antigua and antigua.get("no_regenerar"):
        return f"{ruta_json} (no_regenerar)"

    if antigua:
        relevantes = [c for c, d in antigua["angulos"].items() if d.get("objetivo") is not None or d.get("tol") is not None]
    else:
        relevantes = config.get("relevantes")
    
    relevantes = relevantes or list(ANGULOS["angulos"].keys())

    try:
        contenido = registrar_pose_desde_imagen(
            ruta_imagen=imagen,
            ruta_salida_json=ruta_json,
            nombre_pose=antigua.get("nombre", clave_pose) if antigua else clave_pose,
            articulaciones_relevantes=relevantes,
            mantener_s=antigua.get("mantener_s", 10) if antigua else 10,
            pesos={**PESOS_GENERALES, **config.get("pesos", {})},
        )
    except ValueError as e:
        return f"ERROR {imagen}: {e}"

    angulos = contenido["postura"]["angulos"]
    if antigua: # se respetan la tolerancia y el peso que ya tuviera cada ángulo
        for clave in relevantes:
            for campo in ("tol", "peso"):
                if antigua["angulos"].get(clave, {}).get(campo) is not None:
                    angulos[clave][campo] = antigua["angulos"][clave][campo]
        
        with ruta_json.open("w", encoding="utf-8") as f:
            json.dump(contenido, f, indent=2, ensure_ascii=False)

    lineas = []
    for clave in relevantes:
        valor = "NO EVALUABLE" if angulos[clave]["objetivo"] is None else f"{angulos[clave]['objetivo']:6.1f}"
        antes = antigua["angulos"].get(clave, {}).get("objetivo") if antigua else None
        anterior = f"  (antes {antes})" if antes is not None else ""
        lineas.append(f"    {clave:<20} {valor}  peso={angulos[clave]['peso']}{anterior}")

    sin_medir = [c for c in relevantes if angulos[c]["objetivo"] is None]
    aviso = f"  <-- {len(sin_medir)} sin medir, revisadla" if sin_medir else ""
    estado = "ACTUALIZADA" if antigua else "NUEVA"
    return f"{estado} {ruta_json} <- {imagen}{aviso}\n" + "\n".join(lineas)


def main(poses=None, nuevo=False):
    if poses:
        imagenes = []
        for clave in poses:
            imagen = buscar_imagen(clave)
            if imagen is None:
                print(f"No existe ninguna imagen media/{clave}{{{','.join(EXTENSIONES)}}}")
            else:
                imagenes.append(imagen)
    else:
        imagenes = sorted(p for p in Path("media").glob("*/*") if p.suffix.lower() in EXTENSIONES)

    if not imagenes:
        print("No hay imágenes que procesar")
        return

    for imagen in imagenes:
        print(regenerar(imagen, nuevo=nuevo))

    sin_config = [f"{i.parent.name}/{i.stem}" for i in imagenes if f"{i.parent.name}/{i.stem}" not in CONFIG_POSES]
    if sin_config:
        print("\nSin pesos específicos en CONFIG_POSES (solo PESOS_GENERALES):", ", ".join(sin_config))


if __name__ == "__main__":
    argumentos = sys.argv[1:]
    main([a for a in argumentos if not a.startswith("--")], nuevo="--nuevo" in argumentos) # Puedes indicarle por argumentos exatamente qué imagenes quieres cargar