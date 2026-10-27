# Criterios para juzgar el costo de los modelos en Render

Este documento fija **qué se mide**, **cómo se mide** y **contra qué se decide
si un consumo es aceptable**. Lo implementan `benchmark/stats.py` (resumen y
veredictos) y `benchmark/session.py` (la medición); los umbrales viven en
`benchmark/config.py` y `benchmark/stats.py`.

La precisión NO se juzga acá: eso es `entorno-pruebas`, con los umbrales de
error angular de Gajdosik & Bohannon (≤5° aceptable, 5-10° moderado, >10° no
aceptable). La decisión final de qué modelo va a producción cruza las dos
cosas.

## 1. Qué significa "aceptable" para un costo

La literatura de evaluación de desempeño **no ofrece un umbral universal** de
"CPU aceptable" o "RAM aceptable". Lo que ofrece es un marco y una
metodología:

- **ISO/IEC 25010:2023** define la *eficiencia de desempeño* como el desempeño
  relativo a los recursos usados bajo condiciones establecidas, con tres
  subcaracterísticas: **comportamiento temporal**, **utilización de recursos**
  y **capacidad**. Las tres se definen como el grado en que el sistema
  *cumple los requisitos*. ISO/IEC 25023 da las medidas (tiempo de respuesta,
  tiempo de retorno, rendimiento, utilización de CPU y memoria).
- Por eso **los umbrales salen de los requisitos del servicio desplegado**, no
  de un número genérico. El servicio ya los tiene escritos en su
  configuración de Render (tabla de la sección 3).

Un matiz del dominio: el servicio es **asíncrono**. La plataforma sube el
video, recibe un 200 inmediato y el resultado llega después por webhook. En
telerehabilitación ese es el modo "almacenar y reenviar" (*store-and-forward*;
Russell, 2007), cuya viabilidad clínica está documentada (Bini & Mahajan,
2017). Consecuencia: **no hay requisito de tiempo real**. El factor de tiempo
real se reporta como dato, pero no se usa como criterio de aprobación.

## 2. Métricas

| Métrica (columna) | Qué es | Subcaracterística ISO 25010 | Por qué |
|---|---|---|---|
| `job_s` | `model_init_s` + `processing_s`: construir el modelo y procesar el video completo (decodificar, rotar, convertir, inferir) | Comportamiento temporal | Es lo que paga el worker por cada job: el servicio crea el modelo en cada llamada a `analyze_video`. Ojo: MediaPipe arranca su grafo en el primer `process()`, no al construirse; en él `model_init_s` es casi 0 y ese costo cae en `first_frame_ms` (Heavy: ~320 ms frente a ~98 ms de un frame normal). `job_s` lo incluye igual. |
| `processing_s` | Solo el procesamiento | Comportamiento temporal | Comparable con `elapsed_seconds` de las mediciones en Docker del 2026-08-25. |
| `ms_per_frame`, `fps_processed` | Costo por frame | Comportamiento temporal | El costo escala con los frames; normaliza videos de distinta duración. |
| `realtime_factor` | `processing_s` / duración del video | Comportamiento temporal | Informativo (servicio asíncrono). Permite extrapolar a videos más largos. |
| `infer_ms_p50/p90/p99/max` | Percentiles de la inferencia por frame | Comportamiento temporal | MLPerf usa el percentil 90 como métrica del escenario de un solo flujo (Reddi et al., 2020). |
| `first_frame_ms`, filas `warmup` | Arranque en frío | Comportamiento temporal | El primer job después de un despliegue lo paga. |
| `cpu_cores_used` | Segundos de CPU / segundos de reloj | Utilización de recursos | Exacto (no depende del muestreo); 2,0 = el plan `2c-8g` completo. |
| `cgroup_throttled_pct` | % de periodos en que el kernel frenó al contenedor por pasarse de la cuota | Utilización de recursos | Muestra saturación de CPU aunque el porcentaje medio no la delate. |
| `ram_mb_peak` | Pico exacto de memoria residente del proceso (`VmHWM`, reseteado por corrida) | Utilización de recursos | El riesgo real es un OOM-kill a mitad de un video. |
| `ram_worker_estimate_mb` | `ram_mb_peak` × 2 (el worker corre `--concurrency=2`) | Capacidad | Dos videos a la vez en la misma instancia de 8 GB. |
| `ram_start_mb`, `ram_drift_mb` | RSS al empezar cada job, y cuánto subió entre el primero y el último | Utilización de recursos | Un worker de larga vida que crea un modelo por job no devuelve toda la memoria (ver "Deriva de memoria"). |
| `throughput_videos_per_hour` | Concurrencia × 3600 / `job_s` | Capacidad | Escenario "offline" de MLPerf: lo que importa es cuánto procesa por hora. |
| `max_video_s_within_shutdown` | Segundos de video que caben en 300 s al ritmo medido | Capacidad | `MAX_VIDEO_SIZE_MB` limita megabytes, no duración: esta es la cifra que acota de verdad. |
| `detection_rate` | Fracción de frames con persona detectada | (validez) | Sin persona, MediaPipe se salta la etapa de landmarks: el costo medido sería de otro trabajo. |

## 3. Criterios de aceptabilidad

Se juzga con el **peor caso medido** (`job_s_max`), no con la mediana: un
límite operativo se incumple con una sola corrida que se pase.

### Tiempo (por job)

| Veredicto | Condición | De dónde sale |
|---|---|---|
| **aceptable** | `job_s` ≤ 300 s | `maxShutdownDelaySeconds: 300` del worker en `render.yaml`: al redesplegar, un video que no termina en ese plazo se pierde a la mitad. |
| **condicionado** | 300 s < `job_s` ≤ 900 s | `STUCK_JOB_TIMEOUT_MINUTES=15`: no se marca como atascado, pero cada despliegue puede cortarlo. |
| **no aceptable** | `job_s` > 900 s | El barrido lo da por atascado aunque siga trabajando (y a los 3600 s Redis lo re-entrega: `VISIBILITY_TIMEOUT_SECONDS`). |

### Memoria (del worker completo)

| Veredicto | Condición |
|---|---|
| **aceptable** | 2 × `ram_mb_peak` ≤ 80 % de 8192 MB |
| **condicionado** | hasta 100 % de 8192 MB |
| **no aceptable** | más de 8192 MB (OOM-kill) |

El margen del 20 % **no viene de la literatura**: es una propuesta nuestra para
cubrir lo que el muestreo no ve y el propio proceso de Celery. Hay que
confirmarlo con el equipo.

### Deriva de memoria (hallazgo del 2026-09-28)

Medido en Docker con la imagen de Render (2 CPU / 8 GB), MediaPipe Lite,
12 jobs seguidos de 30 frames de 2EJ1 en el MISMO proceso:

| Cómo se usa el modelo | RSS después de cada job (MB) |
|---|---|
| Uno nuevo por job (lo que hace el servicio) | 250, 283, 360, 439, 441, 418, 445, 469, 417, 419, 481, **559** |
| Uno nuevo por job, con `MALLOC_ARENA_MAX=2` | 228, 290, 301, 302, … , 305, **310** |
| El mismo modelo reutilizado | 278, 285, 255, … , 280, **280** |

No es una fuga de MediaPipe sino fragmentación del asignador de memoria de
glibc: cada `Pose` nuevo crea hilos nuevos, cada hilo puede recibir su propio
*arena*, y dentro del contenedor glibc ve los núcleos del host (12 acá), así
que admite hasta 8 × 12 = 96 arenas antes de estabilizarse. Limitar las
arenas o reutilizar el modelo la elimina.

Qué implica:

- **Para el servicio** (no se toca desde acá): su worker de Celery crea un
  `Pose` por job en un proceso que vive días. Su RAM va a subir con los jobs.
  Las salidas posibles son las tres de la tabla (fijar `MALLOC_ARENA_MAX=2`,
  reutilizar el modelo por proceso, o reciclar procesos con
  `worker_max_tasks_per_child`); elegir una es decisión del servicio.
- **Para esta medición**: se mide como hace el servicio (modelo nuevo por
  repetición, sin `MALLOC_ARENA_MAX`), así que desde la segunda repetición el
  pico incluye lo que dejaron las anteriores. `ram_start_mb` lo deja a la vista
  y `ram_drift_mb` lo resume.
- **Cómo medir la deriva en Render**: muchos jobs cortos en un mismo proceso,
  por ejemplo
  `--models mediapipe_heavy --only 2EJ1 --max-frames 60 --warmup-runs 0 --repetitions 30`,
  y mirar `ram_start_mb` fila por fila.

### CPU

No tiene veredicto propio: el tiempo ya refleja si falta CPU. Se usa para
interpretar. Un modelo con `cpu_cores_used` ≈ 2 (YOLO26-nano en Docker: 198 %)
ocupa toda la instancia: con `--concurrency=2`, dos jobs de YOLO se reparten
los mismos dos núcleos y cada uno tarda más. Uno con ≈ 1 núcleo (MediaPipe) deja
lugar para el segundo job. Por eso la medición final tiene que hacerse también
con `--concurrency 2`.

Señal preliminar (Docker 2 CPU / 8 GB, 90 frames de 5EJ6 y 2EJ1, 2026-09-28;
a confirmar en Render con videos completos):

| Configuración | ms/frame por job | Frames/s del worker |
|---|---|---|
| MediaPipe Heavy, 1 job | ~107 | ~9,3 |
| MediaPipe Heavy, 2 jobs a la vez | ~134 | ~14,9 |
| YOLO26-nano, 1 job, 2 hilos | ~78 a ~107 (CV hasta 26 %: corrida corta y ruidosa) | ~9,3 a ~12,8 |
| YOLO26-nano, 2 jobs a la vez, 2 hilos cada uno | ~199 | ~10,0 |
| YOLO26-nano, 2 jobs a la vez, 1 hilo cada uno | ~145 | ~13,8 |

Con dos jobs a la vez, cada proceso tiene que usar la mitad de los núcleos:
con 2 hilos cada uno, YOLO sobresuscribe la CPU (4 hilos para 2 núcleos) y
el worker procesa ~27 % menos frames por segundo que con 1 hilo cada uno.
MediaPipe Heavy gana ~60 % con dos jobs a la vez; cuánto gana YOLO (ya con
los hilos repartidos) no se puede estimar con estas corridas, porque la de un
solo job fue ruidosa. Es una de las preguntas del paso de medición completa. Por eso `--threads auto` reparte la cuota entre
las sesiones concurrentes (`threads_source = cgroup_quota/2`). Para el
servicio, si algún día sirve YOLO con `--concurrency=2`, aplica lo mismo: sin
fijar hilos, Ultralytics usaría `min(8, os.cpu_count() - 1)`, y en Render
`os.cpu_count()` puede ver los núcleos del host.

### Capacidad

`throughput_videos_per_hour` se compara contra la demanda esperada de la
plataforma. **Ese dato no lo tenemos**: hay que pedirlo (cuántos videos por día
o por hora, y de qué duración).

### Validez de la medición

Una cifra solo entra al documento si:

1. se midió en la instancia `2c-8g` de Render (o, como referencia, en Docker
   con 2 CPU / 8 GB de cgroup), nunca con `--cpu-threads` fuera de un
   contenedor;
2. `versions_match_service` es verdadero (Python 3.12, mediapipe 0.10.14,
   opencv-contrib-python 4.11.0.86, numpy 1.26.4, sin otra distribución de
   OpenCV);
3. hay al menos 3 repeticiones medidas tras 1 de calentamiento;
4. el coeficiente de variación de `processing_s` es ≤ 5 % (si no, repetir con
   más repeticiones: Hoefler & Belli, 2015, recomiendan repetir hasta que el
   intervalo de confianza sea lo bastante estrecho; el 5 % es nuestra elección);
5. `detection_rate` ≥ 50 %.

## 4. Protocolo de medición

- **Calentamiento y repeticiones.** 1 corrida de calentamiento que no se
  analiza (pero se guarda: es el arranque en frío) y 3 medidas (Kalibera &
  Jones, 2013; Georges et al., 2007). Para las cifras finales en Render, 5.
- **Resumen.** Mediana, con desviación estándar, coeficiente de variación e
  intervalo de confianza del 95 % (t de Student). Nunca un número suelto
  (Georges et al., 2007; Hoefler & Belli, 2015).
- **Aislamiento.** Cada modelo y video en su propio proceso: el pico de RAM de
  un proceso que cargó torch no es el de uno que solo cargó MediaPipe.
- **Variabilidad de la nube.** En nubes compartidas el desempeño varía entre
  instancias y a lo largo del día (Leitner & Cito, 2016; Laaber et al., 2019).
  Para que esa variación no caiga siempre sobre el mismo modelo, se usan
  *intentos múltiples intercalados y aleatorizados* (RMIT; Abedi & Brecht,
  2017): `--trials 2 --shuffle-seed <n>`, idealmente en dos momentos
  distintos. Cada fila guarda `cpu_model` y `render_instance_id` para detectar
  si Render cambió de hardware entre corridas.
- **Mismos videos.** El manifiesto fija el SHA-256 de cada video: el portátil,
  Docker y Render miden exactamente el mismo archivo.
- **Rotación incluida.** Los videos se abren con `open_video`, que aplica la
  rotación del celular; ese trabajo queda dentro de `processing_s`, como en el
  servicio.

### Limitaciones que hay que declarar

- `cpu_percent_mean` sale de muestras cada 0,1 s; para decidir se usa
  `cpu_cores_used`, que es exacto.
- `ram_mb_peak` es exacto en Linux (`VmHWM`); en Windows es muestreado
  (`ram_peak_source` lo dice).
- Una medición en Render refleja la instancia que tocó ese día; por eso los
  intentos repetidos.
- Los estudios clínicos de estimación de pose suelen reportar velocidad en
  GPU (p. ej. Rode et al., 2025, en una RTX 4080). El servicio corre sin GPU:
  esas cifras no se pueden trasladar, y es la razón de medir acá.

## 5. Lectura preliminar con las cifras de Docker (2026-08-25)

Solo como orden de magnitud: es Docker Desktop, no Render, un solo video
horizontal (Marcha P01, 334 frames) y un job a la vez.

| Modelo | ms/frame | Núcleos | Video máximo en 300 s (a 29,87 fps) | Video máximo en 900 s |
|---|---|---|---|---|
| MediaPipe Lite | 29 | 1,19 | ≈ 5,7 min | ≈ 17 min |
| MediaPipe Full | 38 | 1,15 | ≈ 4,4 min | ≈ 13 min |
| YOLO26-nano | 75 | 1,98 | ≈ 2,2 min | ≈ 6,7 min |
| MediaPipe Heavy | 106 | 1,05 | ≈ 1,6 min | ≈ 4,8 min |

Lo que sugiere, a verificar en Render:

- Con la tasa de bits de las tomas de TOMA FRONTAL (~2,55 MB/s), el máximo de
  200 MB son ~78 s de video: MediaPipe Heavy tardaría ~250 s, dentro de los
  300 s pero con poco margen, y un celular que grabe con menos tasa de bits
  mete más minutos en los mismos 200 MB.
- YOLO26-nano ocupa los dos núcleos: con dos jobs a la vez, cada uno debería
  tardar cerca del doble. Es la hipótesis que confirma o descarta
  `--concurrency 2`.
- La RAM no es el cuello de botella de ningún modelo (≤ 0,5 GB por proceso).

**Dato que falta del lado de la plataforma**: la duración máxima de los videos
de ejercicio. Con eso se decide si el criterio de 300 s se cumple en el peor
caso real.

## Referencias

- Abedi, A., & Brecht, T. (2017). Conducting repeatable experiments in highly
  variable cloud computing environments. *ICPE '17*.
  https://doi.org/10.1145/3030207.3030229
- Bazarevsky, V., et al. (2020). BlazePose: On-device real-time body pose
  tracking. *arXiv:2006.10204*.
- Bini, S. A., & Mahajan, J. (2017). Clinical outcomes of remote asynchronous
  telerehabilitation are equivalent to traditional therapy following total
  knee arthroplasty. *Journal of Telemedicine and Telecare*.
  https://doi.org/10.1177/1357633X16634518 *(verificar volumen y páginas
  antes de citar)*
- Georges, A., Buytaert, D., & Eeckhout, L. (2007). Statistically rigorous
  Java performance evaluation. *OOPSLA '07*, 57-76.
  https://doi.org/10.1145/1297027.1297033
- Hoefler, T., & Belli, R. (2015). Scientific benchmarking of parallel
  computing systems: Twelve ways to tell the masses when reporting
  performance results. *SC '15*. https://doi.org/10.1145/2807591.2807644
- ISO/IEC 25010:2023. *Systems and software engineering — SQuaRE — Product
  quality model*.
- ISO/IEC 25023:2016. *Systems and software engineering — SQuaRE —
  Measurement of system and software product quality*.
- Kalibera, T., & Jones, R. (2013). Rigorous benchmarking in reasonable time.
  *ISMM '13*. https://doi.org/10.1145/2464157.2464160
- Laaber, C., Scheuner, J., & Leitner, P. (2019). Software microbenchmarking in
  the cloud. How bad is it really? *Empirical Software Engineering, 24*,
  2469-2508. https://doi.org/10.1007/s10664-019-09681-1
- Leitner, P., & Cito, J. (2016). Patterns in the chaos — A study of
  performance variation and predictability in public IaaS clouds. *ACM
  Transactions on Internet Technology, 16*(3). https://doi.org/10.1145/2885497
- Linux kernel. *Control Group v2*.
  https://docs.kernel.org/admin-guide/cgroup-v2.html
- Reddi, V. J., et al. (2020). MLPerf inference benchmark. *ISCA 2020*.
  https://doi.org/10.1109/ISCA45697.2020.00045
- Rode, D., et al. (2025). Assessment of monocular human pose estimation
  models for clinical movement analysis. *Scientific Reports*.
- Russell, T. G. (2007). Physical rehabilitation using telemedicine. *Journal
  of Telemedicine and Telecare, 13*(5), 217-220.
  https://doi.org/10.1258/135763307781458886
