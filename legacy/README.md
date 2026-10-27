# legacy/ — herramientas de la primera fase (historial)

Acá quedan, sin cambios de comportamiento, los scripts y módulos con los que
se produjeron las mediciones anteriores al 2026-09-28:

| Archivo | Produjo |
|---|---|
| `benchmark_yolov8.py` | `results/yolov8_benchmark.csv`, `results/yolov8_docker_2cpu_8gb.csv` |
| `benchmark_mediapipe.py` | `results/mediapipe_docker_2cpu_8gb.csv` |
| `benchmark_yolo_weights.py` | `results/yolo26n_docker_2cpu_8gb.csv` |
| `runner.py`, `resource_monitor.py`, `nvml_process_gpu.py` | el monitor y el runner que usaban los tres |
| `_diag_cpu_limit*.py` | la evidencia del hallazgo del 2026-08-25 (Ultralytics resetea `torch.set_num_threads`). Apuntan a rutas del portátil que ya no existen; se conservan como registro, no para correrlos. |

Se conservan porque hay tablas del documento hechas con esos CSV. **No viajan a
Render** (`.dockerignore` excluye `legacy/`) y no se vuelven a correr salvo
pedido explícito.

Por qué fueron reemplazados por `scripts/run_benchmark.py`:

- Abren el video con `cv2.VideoCapture` directo, sin aplicar la rotación de los
  videos de celular (ver docs/DECISIONES_TECNICAS.md, "Ajustes heredados"). Con Marcha P01
  da igual, porque ese video no trae metadato de rotación, pero con cualquier
  video grabado en vertical miden al modelo trabajando sobre una persona
  acostada.
- Las filas no dicen en qué entorno ni con qué límite de CPU/RAM se midieron,
  ni si la orientación se aplicó.
- Reutilizan el mismo objeto del modelo entre repeticiones, así que MediaPipe
  arrastra el estado de seguimiento de una corrida a la siguiente.
- Solo escriben a un CSV local, que en Render desaparece con el contenedor.

Si hiciera falta volver a correr uno (por ejemplo, para reproducir una tabla):

```powershell
python -m legacy.benchmark_yolov8 --video videos\marcha_p01.mp4 --device cpu --cpu-threads 2 --variants n
```

Los únicos cambios hechos al moverlos fueron los `import` (`benchmark.runner`
pasó a `legacy.runner`, etc.).
