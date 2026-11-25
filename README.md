# Yoga con visión por computador

Juego serio para la Práctica 2 de IPM (Universidad de Alicante) desarrollado por Julio Alfonso De la Torre León y Jorge Izquierdo Baeza. La cámara detecta tu esqueleto con MediaPipe y te puntúa mientras imitas posturas de yoga. La dificultad se elige haciendo una postura, y el juego te avisa con colores y sonidos de si lo estás haciendo bien.

## Cómo ejecutarlo

### 1. Entorno virtual y dependencias

```bash
python3.14 -m venv .venv
source .venv/bin/activate        # En caso de Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

En Linux, para que suene el juego hace falta además `sudo apt install libportaudio2`. Sin él, el juego funciona igual, pero en silencio.

### 2. Descargar los modelos

```bash
python download_models.py
```

Descarga los modelos de MediaPipe en la carpeta `models/`. No están en el repositorio porque pesan bastante.

### 3. Jugar

```bash
python main.py
```

Teclas: `Q` o `ESC` para salir, `M` para volver al menú y `ENTER` en los resultados para volver al menú.

## 3.1 Fragmento en dificultad fácil

https://github.com/user-attachments/assets/3070d477-9951-48d8-abcb-5cdc15382f22

## 3.2 Fragmento en dificultad media

https://github.com/user-attachments/assets/57db2dc6-46bf-49a5-ba8f-a89d4d31e37f

## 3.3 Fragmento en dificultad difícil

https://github.com/user-attachments/assets/0df6e589-8f2a-4c0c-8beb-e50d3e6764dc

### Con Docker (solo Linux)

Antes hay que haber descargado los modelos (paso 2), porque la imagen copia la carpeta `models/` tal cual.

```bash
docker build -t yoga .
xhost +local:docker
docker run --rm -it \
    --device /dev/video0 \
    -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $XDG_RUNTIME_DIR/pulse/native:/tmp/pulse/native \
    yoga
```

Por qué cada cosa:

- `docker build -t yoga .`: crea la imagen con Python, las librerías del sistema que necesitan OpenCV, MediaPipe y el audio, y el código del juego.
- `xhost +local:docker`: da permiso al contenedor para abrir ventanas en tu pantalla.
- `--device /dev/video0`: le presta la cámara al contenedor. (puede ser que tu cámara use otro nombre)
- `-e DISPLAY` y `-v /tmp/.X11-unix`: le dicen en qué pantalla dibujar la ventana del juego.
- `-v .../pulse/native:/tmp/pulse/native`: le presta los altavoces (el `Dockerfile` espera el audio en esa ruta). Sin esta línea el juego funciona, pero sin sonido.
- `--rm -it`: borra el contenedor al cerrar el juego y deja ver los mensajes por la terminal.

La cámara, la pantalla y los altavoces son del ordenador, no del contenedor. Por eso hay que pasárselos al ejecutarlo, y por eso en Windows y macOS es mejor usar el entorno virtual.

## Si no funciona la cámara

El juego usa la cámara con índice `0`, que suele ser la cámara principal del ordenador. Si hay varias cámaras (una externa, una virtual de OBS o Teams, la de infrarrojos de Windows Hello…), puede que la `0` no sea la correcta.

**Síntomas:**

- Al arrancar aparece `No se pudo abrir la cámara, prueba con otro CAMERA_INDEX`.
- Aparece `No se pudo leer un frame de la cámara` y el juego se cierra.
- La ventana se abre, pero la imagen sale negra, congelada o de otra cámara, y el juego nunca te detecta (marco rojo).

**Solución:** cambia el índice en `utils.py` y vuelve a ejecutar:

```python
CAMERA_INDEX = 0 # prueba con 1, 2...
```

Prueba primero con `1` y después con `2`, hasta que se vea tu imagen.

**Otras causas habituales:**

- **La cámara la está usando otra aplicación** (Teams, Zoom, Discord, el navegador…). Ciérrala y vuelve a probar.
- **macOS:** la terminal necesita permiso para usar la cámara. Actívalo en *Ajustes del Sistema → Privacidad y seguridad → Cámara* y vuelve a abrir la terminal.
- **Linux:** con `ls /dev/video*` puedes ver qué cámaras hay. Algunas cámaras crean dos entradas (por ejemplo `video0` y `video1`) y solo una da imagen.
- **Docker:** si tu cámara no es `/dev/video0`, no hace falta cambiar el código: al ejecutar el contenedor, cambia `--device /dev/video0` por, por ejemplo, `--device /dev/video1:/dev/video0`. Así tu cámara aparece dentro del contenedor como la `0`.

## El repositorio

```
yoga/
├── main.py
├── game.py
├── ui.py
├── poses.py
├── selector_dificultad.py
├── sonidos.py
├── suavizado.py
├── utils.py
├── registrador_poses.py
├── regenerador_poses.py
├── tracker_poses.py
├── visualizador_voorherr.py
├── download_models.py
├── media/        fotos de las posturas (facil/, medio/, dificil/)
├── jason/        las posturas de referencia en JSON. Misma estrcutura que media/
├── soniditos/    los sonidos del juego
├── models/       modelos de MediaPipe (se crea con download_models.py)
├── requirements.txt
└── Dockerfile
```

| Archivo | Qué hace |
|---|---|
| `main.py` | Arranca el juego: lee la cámara, detecta el esqueleto y llama a la lógica, la pantalla y los sonidos en cada frame |
| `game.py` | La lógica del juego: menú, explicación, ejecución, nota de cada postura y resultados |
| `ui.py` | Dibuja cada pantalla del juego encima de la imagen de la cámara |
| `poses.py` | Compara tu esqueleto con la postura y calcula la puntuación (también en espejo, para que dé igual el lado) |
| `selector_dificultad.py` | Menú: eliges la dificultad haciendo su postura durante 3 segundos |
| `sonidos.py` | Reproduce los sonidos del juego |
| `suavizado.py` | Quita el temblor y los saltos de los puntos del esqueleto |
| `utils.py` | Configuración (tiempos, umbrales, colores…) y funciones para calcular los ángulos |
| `registrador_poses.py` | Crea el JSON de una postura a partir de una foto |
| `regenerador_poses.py` | Genera los JSON de todas las fotos de `media/` (o solo de las que le digas) |
| `tracker_poses.py` | Herramienta para pruebas: enseña en vivo el valor de cada ángulo |
| `visualizador_voorherr.py` | Herramienta para pruebas: enseña una foto y, al lado, el esqueleto que se detecta en ella |
| `download_models.py` | Descarga los modelos de MediaPipe |
