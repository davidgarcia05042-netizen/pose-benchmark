# Decisiones técnicas — Pose Benchmark en Render

Reúne el contexto del benchmark, las decisiones de diseño con su
justificación, las reglas del código que corre en Render y las convenciones de
trabajo. Cómo instalar, medir y publicar está en el `README.md`; las métricas y
criterios, en `docs/criterios_rendimiento.md`.

## Contexto

Trabajo de grado (Universidad Antonio Nariño, Ingeniería de Sistemas) de Juan
David García Rubio y Camila Sandoval Ruiz, dirigido por Ph.D. David Alberto
Herrera Álvarez, en el marco del grupo de investigación LACSER/Bioingeniería.

Este repositorio mide **cuánto cuesta correr cada modelo de pose** (tiempo,
CPU, RAM) sobre un video. Al principio lo medía en Docker Desktop con límites
de 2 CPU / 8 GB, que simula el presupuesto pero no el procesador físico de
Render.

**Desde el 2026-09-28, todo el código de este repositorio se despliega
directamente en Render.** No hay un paso intermedio de "probar en el portátil
y después adaptar": cada cambio se escribe sabiendo que va a correr en Linux,
dentro de un contenedor de Render, sin nada del disco local.

Lo que este repositorio no mide:

- **Precisión:** eso es [`entorno-pruebas`](https://github.com/davidgarcia05042-netizen/entorno-pruebas).
- **El procesamiento completo del servicio** (validación + análisis +
  ángulos): eso es `app/tools/benchmark_pipeline.py` de
  [`servicio-procesamiento-pose`](https://github.com/davidgarcia05042-netizen/servicio-procesamiento-pose).

Aquí se mide el costo puro del modelo, sin un framework de servicio encima. Por
eso es un repositorio aparte.

## Los tres repositorios de investigación

| Repositorio | Pregunta que responde |
|---|---|
| [`entorno-pruebas`](https://github.com/davidgarcia05042-netizen/entorno-pruebas) | ¿Qué tan preciso es cada modelo contra el gold standard (Maxtraq)? Es la fuente de verdad de qué modelo ganó. No se duplica aquí. |
| [`servicio-procesamiento-pose`](https://github.com/davidgarcia05042-netizen/servicio-procesamiento-pose) | El servicio real: FastAPI + Celery + Redis + PostgreSQL, desplegado en Render. Cuando se diseñó este benchmark, su `render.yaml` ponía el worker en `plan: 2c-8g` (2 CPU, 8 GB), y **ese es el hardware que se mide**. |
| [`pose-benchmark`](https://github.com/davidgarcia05042-netizen/pose-benchmark) (este) | ¿Cuánto cuesta cada modelo en ese hardware? |

## Qué se mide

Los tres modelos activos, decididos en `entorno-pruebas`:

- **YOLO26-nano** (`yolo26n-pose.pt`).
- **MediaPipe Heavy** (`model_complexity=2`) y **MediaPipe Lite**
  (`model_complexity=0`). Se mide también Full, que sirve de referencia aunque
  no es candidato.

Los cuatro se miden con `scripts/run_benchmark.py` (desde el 2026-09-28; ver
"Estructura"). Qué se mide y contra qué se decide si un consumo es aceptable
está en `docs/criterios_rendimiento.md`.

YOLOv8 (n/s/m/l/x) y YOLO11 quedaron descartados. `legacy/benchmark_yolov8.py`
se conserva como historial, porque hay tablas del documento hechas con él. Pero
**no viaja a Render**: no va en la imagen y no se vuelve a correr. Lo mismo vale
para `legacy/benchmark_mediapipe.py` y `legacy/benchmark_yolo_weights.py`, que
produjeron la referencia de Docker de abajo.

El servicio solo sabe servir MediaPipe (`MODEL_COMPLEXITY` en
`app/constants/pose_models.py` del servicio). YOLO26 no tiene código que lo
sirva: es el bloqueador #4 del servicio, y medirlo aquí alimenta esa decisión.

## Punto de partida: lo medido en Docker (2026-08-25)

Condiciones: Docker Desktop con 2 CPU / 8 GB por cgroup, video del conjunto
"Marcha P01" (334 frames, 1920×1080) y 3 repeticiones.

| Variante | Tiempo | CPU medio | RAM pico |
|---|---|---|---|
| MediaPipe Lite | 9,8 s | 119 % | 287 MB |
| MediaPipe Full | 12,6 s | 115 % | 380 MB |
| YOLO26-nano | 24,9 s | 198 % | 403 MB |
| MediaPipe Heavy | 35,3 s | 105 % | 514 MB |

Los datos están en `results/mediapipe_docker_2cpu_8gb.csv` y
`results/yolo26n_docker_2cpu_8gb.csv` (no versionados). YOLO26-nano ocupa los
dos núcleos completos; MediaPipe Heavy tarda más, pero usa uno.

Estos números son la referencia contra la que se compara lo que salga de
Render. No se reemplazan: lo nuevo se agrega al lado, con su origen marcado.

## Resultados en Render (2026-10-01)

Suite TOMA FRONTAL completa (8 videos × 4 modelos, 1 calentamiento + 3
repeticiones), imagen `davidgr05/mvcbenchmark:5a87fa0`, en tres corridas: PC
(Docker 2 CPU / 8 GB, `20261001_toma_frontal_docker_c1`), Render `2c-8g` con
un video a la vez (`20261001T063552Z`) y Render con dos a la vez, como el
worker (`20261001T070703Z`). Todo está en Drive (`pose-benchmark/resultados/`)
y los informes en `pose-benchmark/informes/`: `informe_render_vs_pc.docx`
(`scripts/build_report.py`) e `informe_render_modelos.docx`, solo con Render
(`scripts/build_render_report.py`). El PC compartió el procesador con el uso
del equipo durante su corrida, así que para comparar modelos manda Render.

| Modelo (Render) | ms/frame ×1 | × duración | Núcleos | RAM pico | ms/frame ×2 | Videos/hora ×1 → ×2 | Video máx. en 300 s (×2) |
|---|---|---|---|---|---|---|---|
| YOLO26-nano | 105 | ×3,14 | 1,89 | 422 MB | 127 | 83 → 137 (+65 %) | 74 s |
| MediaPipe Lite | 33 | ×0,98 | 1,43 | 415 MB | 44 | 270 → 387 (+44 %) | 213 s |
| MediaPipe Full | 43 | ×1,28 | 1,33 | 484 MB | 55 | 202 → 312 (+54 %) | 173 s |
| MediaPipe Heavy | 136 | ×4,06 | 1,08 | 649 MB | 142 | 64 → 121 (+88 %) | 67 s |

- Render es un ~26 % más lento por frame que el PC (de +23 % en Full a +30 %
  en YOLO); RAM y núcleos usados casi no cambian.
- Todo es aceptable con estos videos (11–17 s; peor job 72 s). Pero con
  Heavy o YOLO y dos a la vez, un video de más de ~70 s no termina dentro de
  los 300 s del apagado del worker: el servicio limita por MB, no por
  duración.
- Render con 1 a la vez tuvo más variación entre repeticiones (varias
  sesiones con CV > 5 %) que el PC y que Render con 2 a la vez: es una sola
  instancia, conviene repetir en otro momento.
- La repetición 3 de PC/Full/5EJ9 se perturbó por carga en el PC; el control
  `20261001_control_5EJ9_full_docker_c1` la confirma (CV 1,1 %).

## Ajustes heredados del entorno experimental

Se aplicaron **antes** de la primera medición en Render. Estado al
2026-09-28: los tres están aplicados.

### 1. OpenCV no aplica la rotación de los videos de celular

Los celulares graban en vertical guardando los píxeles "acostados", más un
metadato de rotación (por ejemplo 90°) que los reproductores aplican al
mostrar el video. OpenCV no lo aplica por defecto: sin corregirlo, los modelos
ven a la persona acostada.

- **Verificado el 2026-09-28:** `CAP_PROP_ORIENTATION_AUTO` vale 0 al abrir un
  video en los venvs de los tres repositorios (cv2 4.11.0, backend FFMPEG,
  Windows).
- **Efecto medido en `entorno-pruebas`** (TOMA FRONTAL/5EJ6): YOLO26-nano pasó
  de 51,6° a 44,1° de MAE al corregirlo. MediaPipe Heavy no cambió (8,03° →
  8,04°), porque BlazePose alinea la rotación del cuerpo internamente y YOLO
  no.
- Por esto se invalidaron todas las corridas de TOMA FRONTAL anteriores al
  2026-09-28.

Cómo se resolvió aquí:

- **Todo video se abre con una sola función** que fija
  `cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)`, la misma que
  `app/core/video_io.py::open_video` de `entorno-pruebas`:
  `benchmark/video_io.py::open_video`. Los scripts viejos, que abrían con
  `cv2.VideoCapture` directo, quedaron en `legacy/`.
- **No se depende del valor por defecto.** En Windows es 0, y en el contenedor
  Linux de Render no estaba verificado; fijarlo explícito sirve en los dos.
  Verificado el 2026-09-28 con la imagen de este repositorio en Docker (Debian
  13, cv2 4.11.0, cgroup v2): también es 0. Verificado en Render el 2026-10-01
  (cron job `2c-8g`, `python -m benchmark.environment`): también es 0, y
  `open_video` aplica la rotación.
- **La rotación es parte del costo.** Cada frame se rota después de
  decodificarse, y eso es exactamente lo que pasa con los videos que manda la
  plataforma (pacientes grabando con el celular). Queda dentro de la medición,
  no fuera.
- **Cada fila de resultados dice si la orientación se aplicó**, como hace
  `"video_orientation": "metadata_applied"` en el `run_info.json` de
  `entorno-pruebas`, para no mezclar corridas viejas con nuevas. Las corridas
  anteriores con el conjunto Marcha P01 siguen valiendo: ese video no trae
  metadato de rotación.
- **Hace falta al menos un video grabado con celular en vertical.** Con
  Marcha P01 solo, este camino no se ejerce. Verificado el 2026-09-28: los 8
  videos de `suites/toma_frontal.json` declaran rotación de 90° (1920×1080
  almacenado, 1080×1920 al aplicarla), así que lo ejercen. Las pruebas usan un
  MP4 sintético con la rotación parcheada en el átomo `tkhd`
  (`benchmark/synthetic.py`), que no tiene datos de nadie.
- Cada fila guarda `video_orientation`, `video_rotation_meta_deg`,
  `video_stored_resolution` y `video_decoded_resolution`. El build de la
  imagen corre `python -m benchmark.environment --check`, que falla si en ese
  Linux `open_video` no entrega el frame girado.

### 2. El `cv2` que corre no era el que decía requirements.txt

`mediapipe` exige `opencv-contrib-python`, y todas las distribuciones de
OpenCV escriben en el mismo directorio `cv2/`: la que se instala última pisa a
las otras.

- El venv de desarrollo llegó a tener tres instaladas: contrib 4.11.0.86,
  opencv-python 4.11.0.86 y headless 4.10.0.84. El `cv2` efectivo era 4.11.0,
  así que el pin `opencv-python-headless==4.10.0.84` de `requirements.txt` era
  letra muerta.
- El servicio ya lo había resuelto fijando solo
  `opencv-contrib-python==4.11.0.86`, y aquí se hizo igual: se fija la versión
  que efectivamente corre.
- `requirements.txt` fija solo la contrib. Como `ultralytics` exige
  `opencv-python`, el `Dockerfile` la desinstala después y reinstala la contrib
  (en ese orden: al revés, borraría los archivos de la contrib, que son los
  mismos). `benchmark.environment` marca como diferencia con el servicio
  cualquier otra distribución de OpenCV instalada. Los pasos de pip del README
  dejan el venv local igual.

### 3. Las mismas versiones que el servicio en producción

Una medición solo sirve para decidir sobre el servicio si corre las mismas
librerías que el servicio:

- Python 3.12
- mediapipe 0.10.14
- opencv-contrib-python 4.11.0.86
- numpy 1.26.4

Si una de estas versiones cambia en el servicio, cambia aquí también, y los
números anteriores dejan de ser comparables. Están en `requirements.txt` y en
`benchmark/config.py`; cada fila guarda las versiones y
`versions_match_service`. `tests/test_config.py` compara contra el repositorio
del servicio cuando está en la carpeta vecina (también el plan del worker,
`maxShutdownDelaySeconds`, `--concurrency`, `MAX_VIDEO_SIZE_MB`,
`STUCK_JOB_TIMEOUT_MINUTES` y el `visibility_timeout`).

## Reglas del código que va a Render

- **Linux, no Windows.** Se usa `pathlib`, nada de rutas con `\`, y nada que
  dependa del disco del portátil. Se desarrolla en Windows, pero se ejecuta en
  Linux.
- **La imagen lleva solo lo que se mide.** Va `yolo26n-pose.pt`, no los cinco
  pesos de YOLOv8 (~310 MB). `.dockerignore` es una lista blanca
  (`benchmark/`, `scripts/run_benchmark.py`, `suites/`, `yolo26n-pose.pt`,
  `requirements.txt`); `tests/test_config.py` la vigila.
- **Ningún video en la imagen ni en el repositorio.** Aclarado por el equipo el
  2026-09-30: los videos usados (TOMA FRONTAL, Marcha P01) **no son de
  pacientes**; son videos de prueba que entregó el grupo de investigación.
  Igual no viajan en la imagen (que va a un registro, Docker Hub) ni en git: la
  plataforma tendrá videos de pacientes (Ley 1581) y el mecanismo ya queda
  hecho para eso. Los videos se pasan en tiempo de ejecución: con
  `--videos-dir` (en Docker, montados de solo lectura) o bajándolos de Drive
  con `--videos-remote` (`benchmark/storage.py`, rclone). El manifiesto guarda
  su ruta y SHA-256, y el hash se verifica después de bajarlos.
- **El disco de Render es efímero.** Lo que se escribe en `results/`
  desaparece al terminar o al redesplegar. Cada corrida sube
  `results/<run_id>/` a Drive con `--results-remote` (y prueba que puede
  escribir ahí antes de medir). Además, cada fila se imprime en el log
  (`BENCHMARK_ROW {json}`, antes que en el CSV) y `scripts/collect_from_log.py`
  la recupera si la subida falla.
- **Credenciales solo por variables de entorno** (`RCLONE_CONFIG_GDRIVE_*`,
  secretas en Render), nunca en la imagen, el repositorio ni el log. Las
  sesiones de medición no las reciben.
- **`requirements.txt` es exactamente lo que se instala en la imagen.** Lo de
  desarrollo (pytest, lo que usan los `_diag_*`) va en `requirements-dev.txt`,
  igual que en el servicio.
- **No hay GPU.** `nvidia-ml-py` y el monitoreo NVML no aportan nada en Render.
  Pueden quedarse, porque sin GPU dejan esos campos vacíos, pero no se agrega
  nada que dependa de una GPU.
- **Cómo se lee la CPU:**
  - `psutil` reporta el porcentaje relativo a un núcleo, así que 200 % es el
    presupuesto completo de `2c-8g`.
  - `os.cpu_count()` dentro del contenedor puede devolver los núcleos del host
    y no el límite asignado, así que no se usa para calcular porcentajes del
    presupuesto. `benchmark/cgroup.py` lee la cuota real (cgroup v1 y v2) y
    `--threads auto` usa esa. Si Render no deja cuota visible en el cgroup,
    usa `RENDER_CPU_COUNT` (los núcleos del plan, que Render define).
    Verificado en Render el 2026-10-01 (cron job, plan `2c-8g`, Oregon): AMD
    EPYC 7R13 (Zen 3, AWS), cgroup v2 con cuota de 2 núcleos y 8192 MB,
    `os.cpu_count()` = 8, `RENDER_CPU_COUNT` = 2. Es decir, Render limita por
    cuota como Docker; `--threads auto` da 2.
  - Para decidir se usa `cpu_cores_used` (segundos de CPU / segundos de reloj,
    exacto); `cpu_percent_mean` se conserva para comparar con las tablas de
    Docker.
  - Ultralytics fija `NUM_THREADS = min(8, os.cpu_count() - 1)` y lo reaplica
    en la primera inferencia (hallazgo del 2026-08-25). `benchmark/models.py`
    reemplaza esa constante antes de la primera inferencia y reafirma los
    hilos en cada frame; una prueba comprueba que quedan en lo pedido.
- **Lo local no va en la imagen:** el análisis (`scripts/analyze_results.py`,
  sus gráficas y el PDF; `scripts/build_report.py`, el informe Word Render vs.
  PC; `scripts/build_render_report.py`, el informe solo de Render;
  `scripts/charts.py`, lo que comparten), `scripts/collect_from_log.py`, los
  `scripts/_diag_*.py` y `legacy/`.

## Estructura (desde el 2026-09-28)

- `scripts/run_benchmark.py`: punto de entrada. Lanza un proceso por cada
  modelo y video (`benchmark/session.py`), reenvía al log lo que imprimen,
  escribe `results/<run_id>/{rows.csv,summary.csv,run_info.json}` e imprime el
  resumen con el veredicto.
- Cada repetición construye el modelo desde cero (`model_init_s`, como un job
  del servicio) y procesa el video completo (`processing_s`); `job_s` es la
  suma. Las corridas de calentamiento se guardan con `phase=warmup`.
- `python -m benchmark.environment` (comando por defecto de la imagen) reporta
  CPU, límites del cgroup, versiones y si la rotación se aplica, sin videos.
- `docs/criterios_rendimiento.md`: métricas, criterios de aceptación y
  referencias. Los criterios salen de los límites del servicio
  (`benchmark/config.py`).

## Hallazgo: deriva de memoria con un modelo nuevo por job (2026-09-28)

Crear un `Pose` de MediaPipe por job en un proceso de larga vida (lo que hace
el worker del servicio) hace subir la RAM job a job: 250 → 559 MB en 12 jobs
de MediaPipe Lite. Con `MALLOC_ARENA_MAX=2` se estabiliza en ~310 MB y,
reutilizando el modelo, en ~280 MB. Es fragmentación de las arenas de glibc,
no una fuga de MediaPipe. El detalle y la receta para medirlo en Render están
en `docs/criterios_rendimiento.md`, "Deriva de memoria". Aquí se mide igual
que hace el servicio (sin corregirlo) y se registra con `ram_start_mb`; qué
hacer en el servicio es decisión del servicio.

## Bloqueadores abiertos

1. **Plan y facturación de Render.** No los decide el equipo de desarrollo:
   dependen del grupo de investigación. El código no asume un plan, un tipo de
   servicio (worker, cron job, job puntual) ni un costo. El único dato fijo es
   el hardware a medir: el del worker del servicio, `2c-8g`.
2. **Video del tamaño máximo.** Decidido el 2026-09-30: en Render se miden los
   8 videos de TOMA FRONTAL (en vertical, ejercen la rotación), bajados de
   Drive. Pesan 28–43 MB cada uno: sigue faltando uno del tamaño máximo que
   acepta el servicio (`MAX_VIDEO_SIZE_MB` = 200).

Resueltos el 2026-09-30:

- **Dónde quedan los resultados:** en Google Drive (una cuenta personal del
  equipo, provisional mientras el grupo no tenga un espacio propio), carpeta
  `pose-benchmark/resultados/<run_id>/`, compartida con el grupo. Ahí también
  va la carpeta TOMA FRONTAL completa del entorno experimental (2,9 GB).
- **Imagen:** se publica en Docker Hub, repositorio privado
  `davidgr05/mvcbenchmark`, con el commit de git como etiqueta.

## Convenciones del código

- Identificadores en inglés; docstrings y comentarios largos en español (el
  equipo sustenta en español).
- Entorno de desarrollo: Windows, PowerShell, VS Code, Python 3.12, venv por
  proyecto (`venv\Scripts\Activate.ps1`). Entorno de ejecución: el contenedor
  Linux de Render.
- Los scripts se corren como módulo: `python -m scripts.nombre_script`, no
  como archivo suelto.
- Todo cambio de código se verifica con pruebas antes de entregarse: `pytest`
  (suite completa) o `pytest -m "not models"` (sin cargar modelos reales). La
  suite existe desde el 2026-09-28 e incluye
  `test_open_video_applies_metadata_orientation`, equivalente al de
  `entorno-pruebas`. Se verificó también dentro de la imagen de Linux (ver
  README, "Docker local").
- Las cifras de CPU y RAM que van al documento del trabajo de grado salen de
  un límite real: el cgroup de Docker o la instancia de Render, nunca de
  `--cpu-threads` fuera de un contenedor. La razón: Ultralytics restablece
  `torch.set_num_threads` en la primera inferencia (hallazgo del 2026-08-25,
  documentado en `legacy/benchmark_yolov8.py`).
