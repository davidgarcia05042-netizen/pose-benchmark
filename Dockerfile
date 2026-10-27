# Imagen de medición para Render: mide tiempo, CPU y RAM de los modelos de
# pose en el mismo hardware y con las mismas librerías que el worker del
# servicio. Ver README.md, "Despliegue en Render".
#
# Lleva SOLO lo que se mide: el paquete benchmark/, el punto de entrada
# scripts/run_benchmark.py, los manifiestos de suites/ y los pesos de
# yolo26n. Ningún video (se bajan de Drive en cada corrida: la imagen va a un
# registro y los videos son del grupo de investigación), ni los pesos
# descartados de YOLOv8, ni el análisis local. .dockerignore es una lista
# blanca: lo que no está ahí no entra aunque alguien lo copie por error.

# La misma base que el servicio (Debian 13): mismo OpenCV, mismas librerías de
# sistema, mismas condiciones.
FROM python:3.12-slim-trixie

# libgl1 y libglib2.0-0t64: opencv-contrib-python (la variante NO headless, que
# exige mediapipe) carga libGL.so.1 y libgthread al importarse. Igual que en el
# Dockerfile del servicio.
# rclone: baja los videos de Google Drive al arrancar y sube los resultados al
# terminar (benchmark/storage.py). Sus credenciales llegan por variables de
# entorno en tiempo de ejecución, nunca en la imagen.
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0t64 rclone ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# YOLO_OFFLINE / YOLO_AUTOINSTALL: Ultralytics, por defecto, consulta internet,
# envía eventos de uso e intenta instalar con pip lo que le falte. En una
# medición eso es tráfico y tiempo que no son del modelo; en Render, además, una
# dependencia de red escondida. YOLO_CONFIG_DIR: su directorio de ajustes por
# defecto no es escribible sin root. Ultralytics le agrega "/Ultralytics", así
# que con /tmp los ajustes quedan en /tmp/Ultralytics (con /tmp/Ultralytics
# buscaría /tmp/Ultralytics/Ultralytics, caería a su respaldo y avisaría en
# cada sesión de YOLO).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    YOLO_OFFLINE=true \
    YOLO_AUTOINSTALL=false \
    YOLO_CONFIG_DIR=/tmp \
    POSE_BENCHMARK_WEIGHTS_DIR=/app/weights

WORKDIR /app

# 1. torch/torchvision desde el índice CPU (desde PyPI llegaría CUDA).
# 2. El resto, con las versiones fijadas en requirements.txt.
# 3. ultralytics arrastra opencv-python, que pisa el cv2/ de la contrib. Se
#    desinstala y se reinstala la contrib, en ese orden: desinstalar después
#    borraría también los archivos de la contrib, que son los mismos.
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.13.0 torchvision==0.28.0 \
    && pip install -r requirements.txt \
    && pip uninstall -y opencv-python opencv-python-headless \
    && pip install --force-reinstall --no-deps opencv-contrib-python==4.11.0.86

# MediaPipe NO trae los modelos Lite ni Heavy: los descarga dentro de
# site-packages la primera vez que se instancian. Sin root esa descarga
# fallaría, y aunque no fallara sería red y tiempo dentro de la medición. Se
# bajan acá, con la misma función que usa MediaPipe (igual que el servicio).
RUN python -c "from mediapipe.python.solutions.pose import _download_oss_pose_landmark_model as dl; dl(0); dl(2)"

COPY benchmark ./benchmark
COPY scripts/__init__.py scripts/run_benchmark.py ./scripts/
COPY suites ./suites
COPY yolo26n-pose.pt ./weights/

# El commit de git del código medido. Render solo define RENDER_GIT_COMMIT
# cuando construye desde git; con una imagen de Docker Hub el dato tiene que
# venir dentro de la imagen, o cada fila quedaría con git_commit vacío:
#   docker build --build-arg GIT_COMMIT=$(git rev-parse --short HEAD) ...
# Va después de copiar el código para no invalidar la caché de las capas pesadas.
ARG GIT_COMMIT=""
ENV POSE_BENCHMARK_GIT_COMMIT=$GIT_COMMIT

# Falla el build si el cv2 efectivo, mediapipe o numpy no son los del servicio,
# si quedó otra distribución de OpenCV instalada, o si open_video no aplica la
# rotación en este Linux. Mejor enterarse acá que con las cifras ya tomadas.
RUN python -m benchmark.environment --check

# Sin root, igual que el servicio. results/ es de appuser para que el CSV local
# se pueda escribir (el log es el canal que de verdad sale del contenedor).
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/results \
    && chown appuser:appuser /app/results
USER appuser

# Por defecto solo reporta el entorno (CPU, límites del cgroup, versiones,
# orientación): no necesita videos. La medición se lanza con
#   python -m scripts.run_benchmark ...
CMD ["python", "-m", "benchmark.environment"]
