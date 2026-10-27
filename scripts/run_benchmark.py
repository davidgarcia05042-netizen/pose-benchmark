"""
Punto de entrada de la medición: corre cada modelo sobre cada video y deja las
filas en el log y en `results/<run_id>/`.

Cada combinación modelo-video se mide en un proceso aparte
(`benchmark/session.py`), con `--warmup-runs` corridas de calentamiento y
`--repetitions` corridas medidas. Este proceso no carga ningún modelo: solo
reparte el trabajo, reenvía al log lo que imprimen las sesiones, escribe el CSV
y al final imprime el resumen con el juicio de aceptabilidad
(docs/criterios_rendimiento.md).

Uso:

    # Los 8 videos de TOMA FRONTAL, los 4 modelos, 3 repeticiones:
    python -m scripts.run_benchmark --suite suites/toma_frontal.json --videos-dir "<carpeta TOMA FRONTAL>"

    # Solo algunos videos y modelos:
    python -m scripts.run_benchmark --suite suites/toma_frontal.json --videos-dir "<...>" \
        --only 2EJ1 5EJ6 --models yolo26n mediapipe_heavy

    # Dos videos a la vez, como el worker del servicio (--concurrency=2):
    python -m scripts.run_benchmark --suite ... --videos-dir ... --concurrency 2

    # En Render: los videos se bajan de Drive al arrancar y los resultados se
    # suben al terminar (las rutas también pueden venir de las variables
    # POSE_BENCHMARK_VIDEOS_REMOTE y POSE_BENCHMARK_RESULTS_REMOTE):
    python -m scripts.run_benchmark --suite suites/toma_frontal.json \
        --videos-remote "gdrive:pose-benchmark/TOMA FRONTAL" --results-remote "gdrive:pose-benchmark/resultados"

    # Videos sueltos:
    python -m scripts.run_benchmark --video uno.mp4 --video otro.mp4

    # Prueba de humo SIN videos reales (genera uno sintético en vertical):
    python -m scripts.run_benchmark --smoke --repetitions 1 --warmup-runs 0

Metodología de cara a la variabilidad de la nube (Abedi & Brecht, 2017):
`--trials N --shuffle-seed S` repite la suite N veces y baraja el orden
modelo-video en cada intento, para que una racha de ruido del host no caiga
siempre sobre el mismo modelo.

Código de salida: 0 si todo se midió (y se subió, si había --results-remote),
1 si alguna sesión falló, 2 si faltan videos o pesos o no se pudieron bajar los
videos o escribir en --results-remote, 3 si --strict-versions y las versiones
no son las del servicio, 4 si se midió todo pero no se pudieron subir los
resultados (siguen en el log: scripts/collect_from_log.py).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from benchmark import environment, results, stats, storage
from benchmark.models import DEFAULT_MODELS, MODELS
from benchmark.suite import VideoEntry, check_entries, entries_from_paths, load_suite

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS_DIR = Path(os.environ.get("POSE_BENCHMARK_WEIGHTS_DIR", REPO_ROOT))


def build_plan(entries: list[VideoEntry], models: list[str], trials: int, shuffle_seed: int | None) -> list[tuple[int, VideoEntry, str]]:
    """Orden de medición. Sin semilla: video por video, modelo por modelo."""
    base = [(entry, model) for entry in entries for model in models]
    plan = []
    for trial in range(1, trials + 1):
        order = list(base)
        if shuffle_seed is not None:
            random.Random(f"{shuffle_seed}-{trial}").shuffle(order)
        plan.extend((trial, entry, model) for entry, model in order)
    return plan


def session_command(args, entry: VideoEntry, model: str, run_id: str, trial: int, parallel_index: int) -> list[str]:
    command = [
        sys.executable, "-m", "benchmark.session",
        "--model", model,
        "--video", str(entry.path),
        "--video-id", entry.video_id,
        "--repetitions", str(args.repetitions),
        "--warmup-runs", str(args.warmup_runs),
        "--threads", args.threads,
        "--weights-dir", str(args.weights_dir),
        "--sample-interval", str(args.sample_interval),
        "--run-id", run_id,
        "--trial", str(trial),
        "--concurrency", str(args.concurrency),
        "--parallel-index", str(parallel_index),
    ]
    if entry.sha256:
        command += ["--video-sha256", entry.sha256]
    if args.max_frames:
        command += ["--max-frames", str(args.max_frames)]
    return command


class Collector:
    """Recibe las líneas de las sesiones: las reenvía al log y guarda las filas."""

    def __init__(self, sink: results.CsvSink):
        self.sink = sink
        self.rows: list[dict] = []
        self.errors: list[dict] = []
        self._lock = threading.Lock()

    def on_line(self, line: str) -> None:
        with self._lock:
            sys.stdout.write(line if line.endswith("\n") else line + "\n")
            sys.stdout.flush()
            parsed = results.parse_log_lines([line])
            for row in parsed["rows"]:
                self.rows.append(row)
                self.sink.write(row)
            self.errors.extend(parsed["errors"])


def run_sessions(commands: list[list[str]], on_line, timeout_s: float | None, env: dict) -> list[int | None]:
    """
    Lanza las sesiones A LA VEZ y espera a todas. Devuelve el código de salida
    de cada una (None si se cortó por tiempo). Un código negativo en Linux es
    una señal: -9 suele ser el OOM-killer.
    """
    processes = [
        subprocess.Popen(command, cwd=REPO_ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding="utf-8", errors="replace", bufsize=1)
        for command in commands
    ]

    def pump(process):
        for line in process.stdout:
            on_line(line)

    readers = [threading.Thread(target=pump, args=(process,), daemon=True) for process in processes]
    for reader in readers:
        reader.start()
    codes: list[int | None] = []
    # Un solo plazo para todas: esperar en serie con el mismo timeout le daría
    # a la segunda sesión el doble de tiempo.
    deadline = time.monotonic() + timeout_s if timeout_s else None
    try:
        for process in processes:
            try:
                remaining = max(0.0, deadline - time.monotonic()) if deadline else None
                codes.append(process.wait(timeout=remaining))
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                codes.append(None)
    except KeyboardInterrupt:
        for process in processes:
            process.kill()
        raise
    for reader in readers:
        reader.join(timeout=10)
    return codes


def _smoke_entries(directory: Path) -> list[VideoEntry]:
    """Un video sintético 1920x1080 con rotación de 90°, del tamaño de un video de celular."""
    from benchmark.synthetic import write_test_video

    path = write_test_video(directory / "smoke_vertical.mp4", frames=60, width=1920, height=1080, fps=30, rotation=90)
    return [VideoEntry(video_id="smoke_vertical", path=path)]


def _download_videos(rclone: storage.Rclone, remote: str, entries: list[VideoEntry], videos_dir: Path) -> None:
    """Baja los videos del manifiesto. No entra en la medición: el servicio recibe el video por otra vía."""
    files = [entry.path.relative_to(videos_dir).as_posix() for entry in entries]
    print(f"Bajando {len(files)} videos de {remote} a {videos_dir} (no entra en la medición)...", flush=True)
    started = time.monotonic()
    rclone.download(remote, files, videos_dir)
    size_mb = sum(entry.path.stat().st_size for entry in entries if entry.path.is_file()) / (1024 * 1024)
    print(f"Videos listos: {size_mb:.0f} MB en {time.monotonic() - started:.0f} s.", flush=True)


def _upload_results(rclone: storage.Rclone, remote: str, out_dir: Path, run_id: str) -> bool:
    destination = storage.join_remote(remote, run_id)
    try:
        rclone.upload_dir(out_dir, destination)
    except storage.StorageError as error:
        print(f"No se pudieron subir los resultados a {destination}: {error}. "
              "Las filas siguen en este log (scripts/collect_from_log.py).", file=sys.stderr)
        return False
    print(f"Resultados subidos a {destination}", flush=True)
    return True


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m scripts.run_benchmark", description="Mide tiempo, CPU y RAM de los modelos de pose.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--suite", type=Path, help="manifiesto de videos (suites/*.json)")
    source.add_argument("--video", type=Path, action="append", help="un video suelto (se puede repetir)")
    source.add_argument("--smoke", action="store_true", help="usar un video sintético en vertical (sin datos reales)")
    parser.add_argument("--videos-dir", type=Path, help="carpeta base de los archivos del manifiesto")
    parser.add_argument("--videos-remote", default=os.environ.get(storage.VIDEOS_REMOTE_VARIABLE) or None,
                        help="carpeta rclone de donde bajar los videos del manifiesto antes de medir (p. ej. gdrive:...)")
    parser.add_argument("--results-remote", default=os.environ.get(storage.RESULTS_REMOTE_VARIABLE) or None,
                        help="carpeta rclone a la que subir results/<run_id>/ al terminar")
    parser.add_argument("--only", nargs="+", metavar="ID", help="medir solo estos ids del manifiesto")
    parser.add_argument("--models", nargs="+", choices=sorted(MODELS), default=list(DEFAULT_MODELS))
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=1, help="sesiones idénticas a la vez (el worker usa 2)")
    parser.add_argument("--trials", type=int, default=1, help="veces que se repite la suite completa")
    parser.add_argument("--shuffle-seed", type=int, default=None, help="barajar el orden en cada intento (RMIT)")
    parser.add_argument("--threads", default="auto", help="'auto' (del cgroup), un número, o 'library'")
    parser.add_argument("--max-frames", type=int, default=None, help="cortar cada video a N frames (pruebas rápidas)")
    parser.add_argument("--sample-interval", type=float, default=0.1, help="segundos entre muestras de CPU/RAM")
    parser.add_argument("--session-timeout", type=float, default=7200, help="segundos máximos por sesión")
    parser.add_argument("--weights-dir", type=Path, default=DEFAULT_WEIGHTS_DIR)
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--run-id", default=None, help="por defecto, la hora UTC de inicio")
    parser.add_argument("--environment", default=None, help="etiqueta del entorno (por defecto: render/docker/local)")
    parser.add_argument("--skip-hash-check", action="store_true", help="no verificar el SHA-256 del manifiesto")
    parser.add_argument("--strict-versions", action="store_true", help="abortar si las versiones no son las del servicio")
    parser.add_argument("--fail-fast", action="store_true", help="detenerse en la primera sesión que falle")
    return parser


def main(argv: list[str] | None = None) -> int:
    # El log es un entregable (de ahí se recuperan las filas): siempre UTF-8,
    # también cuando en Windows se redirige a un archivo.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = _parser()
    args = parser.parse_args(argv)
    if args.suite and not (args.videos_dir or args.videos_remote):
        parser.error("--suite necesita --videos-dir o --videos-remote")
    if args.videos_remote and not args.suite:
        parser.error("--videos-remote solo aplica con --suite")
    if args.only and not args.suite:
        parser.error("--only solo aplica con --suite")
    if args.repetitions < 1 or args.warmup_runs < 0 or args.concurrency < 1 or args.trials < 1:
        parser.error("--repetitions, --concurrency y --trials deben ser >= 1; --warmup-runs >= 0")

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    child_env = storage.without_secrets(dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8"))
    if args.environment:
        child_env[environment.ENVIRONMENT_OVERRIDE_VARIABLE] = args.environment
        os.environ[environment.ENVIRONMENT_OVERRIDE_VARIABLE] = args.environment

    with tempfile.TemporaryDirectory() as scratch:
        if args.smoke:
            suite_meta, entries = {"name": "smoke"}, _smoke_entries(Path(scratch))
        elif args.suite:
            videos_dir = args.videos_dir or Path(scratch) / "videos"
            suite_meta, entries = load_suite(args.suite, videos_dir, args.only)
        else:
            suite_meta, entries = {"name": "ad_hoc"}, entries_from_paths(args.video)

        rclone = storage.Rclone()
        try:
            if args.videos_remote:
                _download_videos(rclone, args.videos_remote, entries, videos_dir)
            if args.results_remote:
                # Antes de medir: enterarse ahora, y no tras horas de medición,
                # de que las credenciales no sirven.
                rclone.ensure_dir(storage.join_remote(args.results_remote, run_id))
        except storage.StorageError as error:
            print(f"No se puede empezar: {error}", file=sys.stderr)
            return 2

        problems = check_entries(entries, verify_hash=not args.skip_hash_check)
        for model in args.models:
            weights = MODELS[model].weights
            if weights and not (args.weights_dir / weights).is_file():
                problems.append(f"{model}: no están los pesos en {args.weights_dir / weights}")
        if problems:
            print("No se puede empezar:\n  - " + "\n  - ".join(problems), file=sys.stderr)
            return 2

        info = environment.snapshot(include_packages=True)
        info.update(
            run_id=run_id,
            suite=suite_meta.get("name"),
            videos=[{"id": e.video_id, "file": e.path.name, "sha256": e.sha256, "joint": e.joint} for e in entries],
            models=args.models,
            settings={key: value for key, value in vars(args).items() if key not in ("video",)},
        )
        # En el log, sin la lista de paquetes (~100 entradas): alarga mucho la
        # línea y ya va completa en run_info.json, que se sube con los resultados.
        results.emit(results.RUN_PREFIX, {key: value for key, value in info.items() if key != "packages"})
        print(environment.format_report(info), flush=True)
        if info["version_mismatches"] and args.strict_versions:
            print("--strict-versions: las versiones no son las del servicio, no se mide.", file=sys.stderr)
            return 3

        out_dir = args.results_dir / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "run_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        collector = Collector(results.CsvSink(out_dir / "rows.csv"))

        plan = build_plan(entries, args.models, args.trials, args.shuffle_seed)
        failures = []
        for step, (trial, entry, model) in enumerate(plan, 1):
            print(f"\n=== [{step}/{len(plan)}] {model} sobre {entry.video_id} (intento {trial}, concurrencia {args.concurrency}) ===", flush=True)
            commands = [session_command(args, entry, model, run_id, trial, index) for index in range(args.concurrency)]
            codes = run_sessions(commands, collector.on_line, args.session_timeout, child_env)
            for index, code in enumerate(codes):
                if code != 0:
                    reason = "se cortó por tiempo" if code is None else f"salió con {code}" + (" (¿OOM-kill?)" if code == -9 else "")
                    failures.append(f"{model} / {entry.video_id} / proceso {index + 1}: {reason}")
            if failures and args.fail_fast:
                break

    summaries = stats.summarize(collector.rows)
    print("\n=== Resumen (repeticiones medidas; veredicto contra los límites del servicio) ===")
    for summary in summaries:
        results.emit(results.SUMMARY_PREFIX, summary)
        print(stats.format_summary(summary))
    results.write_summary_csv(out_dir / "summary.csv", summaries)
    print(f"\nFilas: {out_dir / 'rows.csv'}  ({len(collector.rows)} filas)")
    uploaded = _upload_results(rclone, args.results_remote, out_dir, run_id) if args.results_remote else True
    if failures:
        print("Sesiones que fallaron:\n  - " + "\n  - ".join(failures), file=sys.stderr)
        return 1
    return 0 if uploaded else 4


if __name__ == "__main__":
    sys.exit(main())
