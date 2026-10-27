"""
Constantes compartidas: el hardware que se mide, las versiones con las que
corre el servicio y los límites operativos del servicio contra los que se
juzga cada medición.

Todo lo que está acá es un dato COPIADO de `../Servicio de procesamiento
asincrono/`. Si allá cambia, tiene que cambiar acá también:
`tests/test_config.py` compara ambos lados cuando el repositorio hermano está
en la misma carpeta (en el portátil), y se salta en Render o en CI.
"""

# --- Hardware a medir: el worker del servicio (render.yaml, plan 2c-8g) ---
RENDER_PLAN = "2c-8g"
BUDGET_CPU_CORES = 2
BUDGET_RAM_MB = 8192

# --- Versiones del servicio en producción (su requirements.txt) ---
# Una medición solo sirve para decidir sobre el servicio si corre las mismas
# librerías. Las claves son nombres de distribución de pip.
SERVICE_PYTHON = (3, 12)
SERVICE_PINNED_VERSIONS = {
    "mediapipe": "0.10.14",
    "opencv-contrib-python": "4.11.0.86",
    "numpy": "1.26.4",
}
# Lo que reporta `cv2.__version__` cuando la distribución efectiva es la fijada.
SERVICE_CV2_VERSION = "4.11.0"
# Distribuciones de OpenCV que NO deben estar instaladas junto a la contrib:
# todas escriben en el mismo directorio cv2/ y la última instalada pisa a las
# otras (ver docs/DECISIONES_TECNICAS.md, "Ajustes heredados", punto 2).
FORBIDDEN_OPENCV_DISTRIBUTIONS = ("opencv-python", "opencv-python-headless", "opencv-contrib-python-headless")

# --- Límites operativos del servicio (render.yaml y app/core/*.py) ---
# `--concurrency=2` del worker: dos videos a la vez en los mismos 2 núcleos.
SERVICE_WORKER_CONCURRENCY = 2
# maxShutdownDelaySeconds del worker: al redesplegar, un video que no termine
# en este plazo se pierde a la mitad.
SERVICE_MAX_SHUTDOWN_DELAY_S = 300
# STUCK_JOB_TIMEOUT_MINUTES: pasado esto, el barrido da el job por atascado
# aunque siga procesando.
SERVICE_STUCK_JOB_TIMEOUT_S = 15 * 60
# VISIBILITY_TIMEOUT_SECONDS del broker: pasado esto, Redis re-entrega la tarea
# y el video se procesa dos veces.
SERVICE_VISIBILITY_TIMEOUT_S = 3600
# MAX_VIDEO_SIZE_MB: tamaño máximo que acepta la API. Ojo: limita megabytes, no
# duración, y el costo de procesar escala con la cantidad de frames.
SERVICE_MAX_VIDEO_SIZE_MB = 200
