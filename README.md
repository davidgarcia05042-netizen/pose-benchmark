# Pose Benchmark — costo de los modelos de pose en Render

Mide **cuánto cuesta correr cada modelo de pose** (tiempo, CPU, RAM) sobre un
video, en el mismo hardware y con las mismas librerías que el worker del
servicio en Render (`plan: 2c-8g`: 2 CPU, 8 GB, sin GPU).

Parte del trabajo de grado de telefisioterapia (Universidad Antonio Nariño,
grupo LACSER/Bioingeniería). Las decisiones de diseño y su justificación están
en [`docs/DECISIONES_TECNICAS.md`](docs/DECISIONES_TECNICAS.md).

| Repositorio | Pregunta |
|---|---|
| [`entorno-pruebas`](https://github.com/davidgarcia05042-netizen/entorno-pruebas) | ¿Qué tan preciso es cada modelo? (fuente de verdad de qué modelo ganó) |
| [`servicio-procesamiento-pose`](https://github.com/davidgarcia05042-netizen/servicio-procesamiento-pose) | El servicio real (FastAPI + Celery + Redis + PostgreSQL) |
| [`pose-benchmark`](https://github.com/davidgarcia05042-netizen/pose-benchmark) (este) | ¿Cuánto cuesta cada modelo en ese hardware? |

La plataforma que usa el servicio: [`telefisioterapia-backend`](https://github.com/davidgarcia05042-netizen/telefisioterapia-backend),
[`telefisioterapia-panel`](https://github.com/davidgarcia05042-netizen/telefisioterapia-panel) y
[`telefisioterapia-app`](https://github.com/davidgarcia05042-netizen/telefisioterapia-app).

Acá se mide el costo puro del modelo, sin el framework del servicio encima. El
pipeline completo del servicio se mide con su `app/tools/benchmark_pipeline.py`.

Modelos: **YOLO26-nano**, **MediaPipe Heavy** y **MediaPipe Lite** (los
candidatos) y **MediaPipe Full** (referencia).

## Estructura

```
benchmark/            paquete de medición (va a la imagen)
  video_io.py           open_video: aplica la rotación de los videos de celular
  session.py            UN modelo sobre UN video, en su propio proceso
  models.py             cómo se corre cada modelo (y la corrección de hilos de YOLO)
  monitor.py            CPU (segundos de CPU) y RAM (pico exacto VmHWM) por corrida
  cgroup.py             límites y contadores reales del contenedor (v1 y v2)
  environment.py        entorno, versiones vs. el servicio; `python -m benchmark.environment`
  results.py            esquema de filas; salen por el log (BENCHMARK_ROW) y a CSV
  stats.py              resumen estadístico y veredicto contra los límites del servicio
  storage.py            baja los videos de Drive y sube los resultados (rclone)
  suite.py, synthetic.py  manifiestos de videos; videos sintéticos para tests
scripts/
  run_benchmark.py      punto de entrada de la medición (va a la imagen)
  collect_from_log.py   reconstruye resultados desde un log de Render (local)
  analyze_results.py    resumen, gráficas y PDF de una o más corridas (local)
  build_report.py       informe Word Render vs. PC: tablas, gráficas, diferencias (local)
  build_render_report.py  informe Word solo de Render: comparación entre modelos (local)
  charts.py             gráficas y formato compartidos por los dos anteriores (local)
  _diag_*.py            figuras puntuales del documento, con cifras fijas (local)
suites/toma_frontal.json  los 8 videos de TOMA FRONTAL, con su SHA-256
docs/criterios_rendimiento.md  métricas, criterios de aceptación y referencias
legacy/               scripts de la primera fase (historial, no van a Render)
tests/                suite de pytest
```

## Instalación local y tests

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.13.0 torchvision==0.28.0
pip install -r requirements-dev.txt
pip uninstall -y opencv-python opencv-python-headless
pip install --force-reinstall --no-deps opencv-contrib-python==4.11.0.86

pytest                    # todo
pytest -m "not models"    # sin cargar modelos reales (segundos)
```

Los dos últimos pasos de pip dejan una sola distribución de OpenCV, la contrib,
que es la que corre el servicio. `python -m benchmark.environment` dice si el
entorno quedó comparable con el servicio.

## Medir

**Los videos no están en el repositorio y su ruta es configurable.** Se pasan
en tiempo de ejecución con `--videos-dir <carpeta>` (o `VIDEOS_DIR` en Docker)
o se bajan de Drive con `--videos-remote` / `POSE_BENCHMARK_VIDEOS_REMOTE`. La
carpeta debe tener la estructura del manifiesto (`suites/toma_frontal.json`:
`<id>/<id>.mp4`), y el SHA-256 de cada video se verifica antes de medir. En
los ejemplos, `<datos>` es la carpeta local del gold standard (en el equipo de
desarrollo, `data/gold_standard` de `entorno-pruebas`).

```powershell
# Los 8 videos de TOMA FRONTAL, los 4 modelos, 1 calentamiento + 3 repeticiones:
python -m scripts.run_benchmark --suite suites/toma_frontal.json `
    --videos-dir "<datos>\TOMA FRONTAL"

# Solo algunos videos o modelos:
python -m scripts.run_benchmark --suite suites/toma_frontal.json --videos-dir "<...>" `
    --only 2EJ1 5EJ6 --models yolo26n mediapipe_heavy

# Dos videos a la vez, como el worker (--concurrency=2):
python -m scripts.run_benchmark --suite ... --videos-dir ... --concurrency 2

# Contra la variabilidad de la nube: 2 intentos con el orden barajado (RMIT):
python -m scripts.run_benchmark --suite ... --videos-dir ... --trials 2 --shuffle-seed 1

# Prueba de humo sin ningún video real (genera uno sintético en vertical):
python -m scripts.run_benchmark --smoke --repetitions 1 --warmup-runs 0
```

> **Las cifras solo valen en un límite real**: la instancia `2c-8g` de Render o,
> como referencia, Docker con 2 CPU / 8 GB. Correrlo en el portátil sirve para
> probar la herramienta, no para el documento.

Cada corrida deja en `results/<run_id>/`:

- `rows.csv`: una fila por repetición (78 columnas: tiempos, percentiles por
  frame, CPU, RAM, límites del cgroup, rotación del video, versiones). Las
  filas `phase=warmup` son el arranque en frío y no entran al análisis.
- `summary.csv`: mediana, dispersión y veredicto por modelo y video.
- `run_info.json`: el entorno completo, incluida la lista de paquetes.

Y **cada fila sale también por el log** como `BENCHMARK_ROW {json}`: en Render
el disco es efímero y el log es lo que sobrevive (ver "Despliegue en Render").

## Qué se considera aceptable

Resumen de `docs/criterios_rendimiento.md` (ahí están las métricas, el
protocolo y las referencias):

| Criterio | Aceptable | Condicionado | No aceptable |
|---|---|---|---|
| Tiempo por job (peor caso) | ≤ 300 s (plazo de apagado del worker) | ≤ 900 s (umbral de job atascado) | > 900 s |
| RAM del worker (2 × pico) | ≤ 80 % de 8 GB | ≤ 8 GB | > 8 GB |

Además, una medición solo es válida con ≥ 3 repeticiones, coeficiente de
variación ≤ 5 %, detección de la persona ≥ 50 % y las mismas versiones que el
servicio.

## Analizar

```powershell
python -m scripts.analyze_results --csv results\<run_id>\rows.csv --out results\<run_id>\analisis

# Lo nuevo al lado de la referencia de Docker del 2026-08-25:
python -m scripts.analyze_results --csv results\<run_id>\rows.csv `
    --legacy-csv results\mediapipe_docker_2cpu_8gb.csv results\yolo26n_docker_2cpu_8gb.csv `
    --out results\comparacion
```

### Informe Render vs. PC (Word)

La misma suite medida en Render y en el PC (Docker con 2 CPU / 8 GB), lado a
lado: entornos, videos, protocolo, criterios, resultados por modelo, tiempo
por job frente a los límites del servicio, CPU, memoria, diferencias
documentadas, limitaciones y anexos con todas las filas resumidas.

```powershell
# 1. Traer de Drive la corrida de Render (la sube el propio cron job):
rclone lsf gdrive:pose-benchmark/resultados/
rclone copy gdrive:pose-benchmark/resultados/<run_id_render> results\<run_id_render>

# 2. Armar el informe (las gráficas quedan en <nombre>_figuras/):
python -m scripts.build_report --pc results\<run_id_pc> --render results\<run_id_render> `
    --out results\informes\informe_render_vs_pc.docx --image-tag davidgr05/mvcbenchmark:<tag>

# 3. Compartirlo con el grupo:
rclone copy results\informes gdrive:pose-benchmark/informes
```

Se puede generar con una corrida a medias o sin la de Render: el informe dice
cuántas sesiones tiene cada una y deja la comparación pendiente. Solo usa
sesiones modelo-video con las 3 repeticiones completas. `--render-c2` agrega
la corrida con dos videos a la vez y `--note` deja constancia de algo que pasó
durante una corrida, sin tocar los datos.

### Informe solo de Render: comparación entre modelos (Word)

Sin el PC (que comparte el procesador con lo que se use en el equipo): qué
modelo es más eficiente en el hardware del worker, su consumo de CPU y RAM,
cuánto rinde con dos videos a la vez, qué duración de video aguanta y un
indicador de costo por video.

```powershell
python -m scripts.build_render_report --render results\<run_id_c1> --render-c2 results\<run_id_c2> `
    --out results\informes\informe_render_modelos.docx --image-tag davidgr05/mvcbenchmark:<tag>
```

## Docker local (réplica de 2 CPU / 8 GB)

```powershell
docker compose build
docker compose run --rm benchmark      # reporte: CPU, límites del cgroup, versiones, rotación

$env:VIDEOS_DIR = "<datos>\TOMA FRONTAL"
docker compose run --rm benchmark python -m scripts.run_benchmark `
    --suite suites/toma_frontal.json --videos-dir /data/videos
```

Los videos se montan de solo lectura desde `VIDEOS_DIR`; nunca entran a la
imagen (`.dockerignore` es una lista blanca). Las filas quedan etiquetadas
`docker_2cpu_8gb`.

La suite de tests también corre en el Linux de la imagen (se monta la carpeta
padre para que corran las comparaciones contra el servicio):

```powershell
docker run --rm -u root -v "$((Resolve-Path ..).Path):/work:ro" -w /work/pose-benchmark `
    pose-benchmark:render sh -c "pip install -q pytest==9.1.1 && python -m pytest -q -p no:cacheprovider"
```

Ojo: la VM de WSL 2 tiene su propio tope de memoria (en este portátil,
~7,8 GB). Si es menor que 8 GB, el `mem_limit: 8g` no se alcanza. Para estos
modelos no importa (usan < 1 GB), pero hay que saberlo antes de afirmar que
se midió "con 8 GB".

## Despliegue en Render

Lo que ya está listo:

- La imagen (`Dockerfile`) usa la misma base que el servicio, instala las
  mismas versiones, deja una sola distribución de OpenCV, descarga los modelos
  Lite y Heavy de MediaPipe al construir, corre sin root y **falla el build**
  si las versiones no son las del servicio o si la rotación no se aplica.
- El comando por defecto de la imagen es `python -m benchmark.environment`: no
  necesita videos y confirma qué CPU, qué límites de cgroup y qué versiones da
  de verdad la instancia. Es un buen primer despliegue.
- `--threads auto` toma los núcleos de la cuota del cgroup, no los del host.
- Los videos se bajan de Google Drive al arrancar y `results/<run_id>/` se
  sube a Drive al terminar (ver abajo). Además, cada fila sale por el log: si
  la subida falla (código de salida 4), `python -m scripts.collect_from_log
  render.log` reconstruye `results/` desde el log descargado. Render conserva
  los logs por tiempo limitado.
- La imagen se publica en Docker Hub, en el repositorio privado
  `davidgr05/mvcbenchmark`, etiquetada con el commit de git (no `latest`), para
  saber qué código produjo cada medición.

Lo que está pendiente (bloqueador #1, ver `docs/DECISIONES_TECNICAS.md`): el plan y el tipo de
servicio en Render. No hay una solución asumida en el código.

## Videos y resultados en Google Drive

La imagen no lleva videos. Cada corrida baja de Drive los del manifiesto,
verifica su SHA-256 y, al terminar, sube sus resultados a la misma carpeta que
ve el grupo de investigación. El transporte es [rclone](https://rclone.org)
(`benchmark/storage.py`); la descarga ocurre antes de medir y no entra en
ningún tiempo.

Estructura en Drive:

```
pose-benchmark/
├── TOMA FRONTAL/          copia completa de data/gold_standard/TOMA FRONTAL (entorno-pruebas)
│   ├── 2EJ1/2EJ1.mp4      (videos, resultados de precisión, videos de comparación)
│   └── ...
└── resultados/<run_id>/   rows.csv, summary.csv, run_info.json de cada corrida
```

### Una sola vez, desde el portátil

1. **Client ID propio en Google Cloud** (el compartido de rclone deja de
   funcionar en 2026), creado con la cuenta institucional: proyecto nuevo →
   habilitar Google Drive API → Google Auth Platform: Audience **Internal**
   (con External en modo "Testing" el token vence a los 7 días) → Data access:
   `.../auth/drive.file` → Clients: tipo **Desktop app**.
2. **rclone en el portátil.** `winget install Rclone.Rclone` no siempre lo
   agrega al PATH; si `rclone` no se reconoce, agregar a mano la carpeta
   `%LOCALAPPDATA%\Microsoft\WinGet\Packages\Rclone.Rclone_*\rclone-v*-windows-amd64`.
3. **Autorizar** con la cuenta institucional. Si el navegador abre otra
   cuenta, cerrar esa pestaña y abrir en incógnito el enlace
   `http://127.0.0.1:53682/auth?state=...` que rclone imprime en la terminal.
   `scope=drive.file`: rclone solo ve lo que él mismo sube, no el resto del
   Drive.
4. **Subir TOMA FRONTAL** con la ruta absoluta (una relativa depende de la
   carpeta en que esté la terminal).

```powershell
rclone config create gdrive drive scope=drive.file client_id=<CLIENT_ID> client_secret=<CLIENT_SECRET>
rclone mkdir "gdrive:pose-benchmark"
$origen = "C:\...\<datos>\TOMA FRONTAL"
rclone copy $origen "gdrive:pose-benchmark/TOMA FRONTAL" --progress
rclone check $origen "gdrive:pose-benchmark/TOMA FRONTAL"   # 0 differences, sin ERROR
```

El token (`rclone config show gdrive`) es una contraseña: no se pega en chats,
issues ni archivos del repositorio. Si se filtra, se revoca el acceso de la app
en https://myaccount.google.com/permissions y se genera otro con
`rclone config reconnect gdrive:`.

Luego se comparte la carpeta `pose-benchmark` con el grupo desde la web de
Drive (como lector). Con `scope=drive.file`, lo que otros suban por la web a
esa carpeta no lo ve rclone: los videos que se van a medir se suben con rclone.

### Variables de entorno en Render (secretas)

`rclone config show gdrive` muestra el token. En Render:

| Variable | Valor |
|---|---|
| `RCLONE_CONFIG_GDRIVE_TYPE` | `drive` |
| `RCLONE_CONFIG_GDRIVE_SCOPE` | `drive.file` |
| `RCLONE_CONFIG_GDRIVE_CLIENT_ID` | el client ID de Google Cloud |
| `RCLONE_CONFIG_GDRIVE_CLIENT_SECRET` | el client secret |
| `RCLONE_CONFIG_GDRIVE_TOKEN` | el JSON del token, completo |
| `POSE_BENCHMARK_VIDEOS_REMOTE` | `gdrive:pose-benchmark/TOMA FRONTAL` |
| `POSE_BENCHMARK_RESULTS_REMOTE` | `gdrive:pose-benchmark/resultados` |

Con eso, el comando de la corrida en Render es:

```
python -m scripts.run_benchmark --suite suites/toma_frontal.json
```

Si las credenciales no sirven, la corrida se detiene **antes** de medir
(código 2): prueba a crear `resultados/<run_id>/` en Drive al arrancar. Las
sesiones de medición no reciben las variables `RCLONE_CONFIG_*`.

### Probar el circuito completo en Docker local

Antes de Render, con la misma imagen y las mismas variables. Las credenciales
se toman de la configuración local de rclone y pasan como variables de la
sesión de PowerShell (`-e NOMBRE` sin valor las hereda): así el token no queda
en ningún archivo ni en el historial, y PowerShell 5.1 no le borra las
comillas al JSON, como haría con `-e NOMBRE=valor`.

```powershell
$cfg = (rclone config dump | ConvertFrom-Json).gdrive
$env:RCLONE_CONFIG_GDRIVE_TYPE = "drive"
$env:RCLONE_CONFIG_GDRIVE_SCOPE = "drive.file"
$env:RCLONE_CONFIG_GDRIVE_CLIENT_ID = $cfg.client_id
$env:RCLONE_CONFIG_GDRIVE_CLIENT_SECRET = $cfg.client_secret
$env:RCLONE_CONFIG_GDRIVE_TOKEN = $cfg.token
docker run --rm --cpus 2 --memory 8g --memory-swap 8g `
    -e RCLONE_CONFIG_GDRIVE_TYPE -e RCLONE_CONFIG_GDRIVE_SCOPE -e RCLONE_CONFIG_GDRIVE_CLIENT_ID `
    -e RCLONE_CONFIG_GDRIVE_CLIENT_SECRET -e RCLONE_CONFIG_GDRIVE_TOKEN `
    -e POSE_BENCHMARK_ENVIRONMENT=docker_2cpu_8gb `
    -e "POSE_BENCHMARK_VIDEOS_REMOTE=gdrive:pose-benchmark/TOMA FRONTAL" `
    -e POSE_BENCHMARK_RESULTS_REMOTE=gdrive:pose-benchmark/resultados `
    pose-benchmark:render python -m scripts.run_benchmark --suite suites/toma_frontal.json `
        --only 2EJ1 --models mediapipe_lite --repetitions 1 --warmup-runs 0 --max-frames 60
```

Tiene que terminar con `Resultados subidos a gdrive:pose-benchmark/resultados/<run_id>`.

## Publicar la imagen en Docker Hub

Repositorio privado `davidgr05/mvcbenchmark`. La etiqueta es el commit de
git, para saber qué código produjo cada medición (`latest` no lo dice). Se
publica solo desde un árbol limpio (todo commiteado):

```powershell
docker login
$tag = git rev-parse --short HEAD
docker build --build-arg GIT_COMMIT=$tag -t davidgr05/mvcbenchmark:$tag .
docker push davidgr05/mvcbenchmark:$tag
```

En Render, el servicio apunta a `docker.io/davidgr05/mvcbenchmark:<tag>` con
una credencial de registro (un access token de Docker Hub de solo lectura).

## legacy/

Los scripts de la primera fase (YOLOv8, MediaPipe y YOLO26 en Docker,
2026-08-25) quedan en `legacy/` sin cambios de comportamiento, porque hay
tablas del documento hechas con ellos. Ver `legacy/README.md`.
