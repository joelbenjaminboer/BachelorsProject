"""Per-activity RMSE for the Transformer encoder (fold AB156).

Reads eval_encoder_final.json (produced by src/training/eval.py).
"""

import json

import matplotlib.pyplot as plt

with open("eval_encoder_final.json") as f:
    data = json.load(f)

activities = ["walk", "ramp_up", "ramp_down", "stair_up", "stair_down"]
labels = ["Walk", "Ramp Up", "Ramp Down", "Stair Up", "Stair Down"]
rmse = [data["activity_metrics"][a]["rmse"] for a in activities]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(labels, rmse, color="#d62728")
ax.set_ylabel("RMSE (°)")
ax.set_title("Transformer encoder RMSE by activity (fold AB156)")
ax.set_ylim(0, max(rmse) * 1.2)
ax.grid(axis="y", alpha=0.3)

for bar, val in zip(bars, rmse):
    ax.annotate(
        f"{val:.2f}",
        (bar.get_x() + bar.get_width() / 2, val),
        ha="center",
        va="bottom",
        xytext=(0, 4),
        textcoords="offset points",
    )

plt.tight_layout()
plt.savefig("img/activity_rmse.png", dpi=200)
print("Plot successfully saved to img/activity_rmse.png")
