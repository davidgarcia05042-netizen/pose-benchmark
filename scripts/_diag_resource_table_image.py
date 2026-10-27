import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = [
    ("MediaPipe Lite", "9.8s", "119.2%", "289 MB", "9.43°"),
    ("MediaPipe Full", "12.6s", "115.0%", "380 MB", "10.94°"),
    ("YOLO26-nano", "24.9s", "198.3%", "404 MB", "8.45°"),
    ("MediaPipe Heavy", "35.3s", "105.5%", "514 MB", "7.85°"),
]
columns = ["Variante", "Tiempo", "CPU medio", "RAM pico", "Precisión\n(error medio)"]

fig, ax = plt.subplots(figsize=(11, 3.4))
ax.axis("off")

table = ax.table(
    cellText=rows,
    colLabels=columns,
    cellLoc="center",
    loc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 1.9)

row_colors = {
    "YOLO26-nano": "#dbeafe",
    "MediaPipe Heavy": "#fee2e2",
}
for i, row in enumerate(rows, start=1):
    variant = row[0]
    color = row_colors.get(variant, "#ffffff")
    for j in range(len(columns)):
        table[i, j].set_facecolor(color)

for j in range(len(columns)):
    table[0, j].set_facecolor("#1f2937")
    table[0, j].set_text_props(color="white", fontweight="bold")

ax.set_title(
    "Recursos reales (contenedor Docker, 2 CPU / 8GB) y precisión -- video Marcha P01",
    fontsize=12,
    pad=15,
)
plt.tight_layout()
plt.savefig("results/resource_comparison_table.png", dpi=150, bbox_inches="tight")
print("guardado")
