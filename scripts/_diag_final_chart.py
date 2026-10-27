import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

labels = ["MediaPipe\nLite", "MediaPipe\nFull", "YOLO26\nnano", "MediaPipe\nHeavy"]
times = [9.8, 12.6, 24.9, 35.3]
cpu = [119.2, 115.0, 198.3, 105.5]
ram = [289, 380, 404, 514]
accuracy = [9.43, 10.94, 8.45, 7.85]
colors = ["#fca5a5", "#ef4444", "#2563eb", "#b91c1c"]

fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))

for ax, values, title, ylabel, ref_lines in [
    (axes[0], times, "Tiempo (s)", "segundos", None),
    (axes[1], cpu, "CPU medio (%)", "% (200%=2 nucleos)", [200]),
    (axes[2], ram, "RAM pico (MB)", "MB", None),
    (axes[3], accuracy, "Error angular (grados)", "grados", [5, 10]),
]:
    ax.bar(labels, values, color=colors)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if ref_lines:
        for rl in ref_lines:
            ax.axhline(rl, color="gray", linestyle="--", alpha=0.5)
    ax.tick_params(axis="x", labelsize=8)

fig.suptitle("YOLO26-nano vs. MediaPipe (Lite/Full/Heavy) -- recursos reales en contenedor 2 CPU / 8GB", fontsize=12)
plt.tight_layout()
plt.savefig("results/final_comparison_chart.png", dpi=120)
print("guardado")
