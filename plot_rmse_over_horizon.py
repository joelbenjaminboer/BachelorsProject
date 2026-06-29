"""Per-step RMSE across the forecast horizon for the Transformer encoder
under the main LOSO protocol.

Reads eval_encoder_final.json (produced by src/training/eval.py). TCN's
per-step series is not yet available; once benchmarked with a matching eval
JSON structure, add a second curve here.
"""

import json

import matplotlib.pyplot as plt
import numpy as np

with open("eval_encoder_final.json") as f:
    data = json.load(f)

rmse = np.array(data["per_step"]["rmse"])
steps = np.arange(1, len(rmse) + 1)
target_freq = 100
t_sec = steps / target_freq

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(t_sec, rmse, color="#d62728", linewidth=1.8, label="Transformer encoder")
ax.set_xlabel("Time into forecast window (s)")
ax.set_ylabel("RMSE (°)")
ax.set_title("Per-step RMSE across the forecast horizon (LOSO, fold AB156)")
ax.grid(alpha=0.3)
ax.legend(loc="lower right")
ax.set_xlim(t_sec[0], t_sec[-1])

plt.tight_layout()
plt.savefig("img/rmse_over_horizon.png", dpi=200)
print("Plot successfully saved to img/rmse_over_horizon.png")
