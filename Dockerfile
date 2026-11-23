FROM python:3.14-slim

# Librerías del sistema:
#  - OpenCV (GUI con Qt/X11) + fuentes para Qt
#  - MediaPipe (OpenGL, EGL, GLES)
#  - Audio: PortAudio (sounddevice) + plugin ALSA->PulseAudio
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libegl1 libgles2 libglib2.0-0 \
        libx11-6 libxext6 libxrender1 libsm6 libice6 \
        libxcb1 libxkbcommon0 libxkbcommon-x11-0 \
        libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 \
        libxcb-render-util0 libxcb-shape0 libxcb-xinerama0 libxcb-xkb1 \
        libxcb-xfixes0 libxcb-cursor0 \
        libfontconfig1 fonts-dejavu-core libdbus-1-3 \
        libportaudio2 libasound2-plugins \
    && rm -rf /var/lib/apt/lists/*

# ALSA usa PulseAudio (el socket del host) como dispositivo por defecto
RUN printf 'pcm.!default { type pulse }\nctl.!default { type pulse }\n' > /etc/asound.conf

ENV PYTHONUNBUFFERED=1
ENV QT_X11_NO_MITSHM=1
ENV QT_QPA_FONTDIR=/usr/share/fonts/truetype/dejavu
ENV PULSE_SERVER=unix:/tmp/pulse/native

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "main.py"]