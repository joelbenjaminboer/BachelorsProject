"""Illustrates the stride-1 sliding-window scheme: a context window of IMU
samples (top) paired with the corresponding forecast window of goniometer
samples (bottom), shown at several overlapping starting positions.

Each window is drawn as a horizontal bracket beneath/above its respective
signal, offset vertically so that heavily overlapping windows remain
distinguishable, rather than as stacked translucent fills.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FILE_PATH = "data/raw/ENABL3S/AB185/Processed/AB185_Circuit_003_post.csv"
ORIGINAL_FREQ = 500
TARGET_FREQ = 100
DOWNSAMPLE = ORIGINAL_FREQ // TARGET_FREQ

CONTEXT_LENGTH = 274
FORECAST_HORIZON = 137

# Stride between successive illustrated windows (in samples, at 100 Hz).
# A larger value is used here purely so the overlapping windows remain
# visually distinguishable; the real preprocessing uses stride = 1.
DISPLAY_STRIDE = 137
N_WINDOWS = 3

ACCEL_CHANNELS = ["Right_Thigh_Ax", "Right_Thigh_Ay", "Right_Thigh_Az"]
GYRO_CHANNELS = ["Right_Thigh_Gx", "Right_Thigh_Gy", "Right_Thigh_Gz"]

df = pd.read_csv(FILE_PATH)
accel = {c: df[c].to_numpy()[::DOWNSAMPLE] for c in ACCEL_CHANNELS}
gyro = {c: df[c].to_numpy()[::DOWNSAMPLE] for c in GYRO_CHANNELS}
imu = accel["Right_Thigh_Ay"]
gonio = df["Right_Knee"].to_numpy()[::DOWNSAMPLE]

start0 = 200
t = np.arange(start0, start0 + CONTEXT_LENGTH + (N_WINDOWS - 1) * DISPLAY_STRIDE + FORECAST_HORIZON)
t_sec = t / TARGET_FREQ

colors = plt.cm.viridis(np.linspace(0.1, 0.85, N_WINDOWS))

fig, (ax_imu, ax_gonio) = plt.subplots(
    2,
    1,
    figsize=(11, 7.2),
    gridspec_kw={"hspace": 0.35},
)

accel_colors = {"Right_Thigh_Ax": "#1f77b4", "Right_Thigh_Ay": "#d62728", "Right_Thigh_Az": "#2ca02c"}
gyro_colors = {"Right_Thigh_Gx": "#1f77b4", "Right_Thigh_Gy": "#d62728", "Right_Thigh_Gz": "#2ca02c"}

ax_gyro = ax_imu.twinx()

for c in ACCEL_CHANNELS:
    ax_imu.plot(
        t_sec,
        accel[c][t[0] : t[-1] + 1],
        color=accel_colors[c],
        linewidth=1.1,
        zorder=3,
        label=c.split("_")[-1],
    )
for c in GYRO_CHANNELS:
    ax_gyro.plot(
        t_sec,
        gyro[c][t[0] : t[-1] + 1],
        color=gyro_colors[c],
        linewidth=1.0,
        linestyle="--",
        alpha=0.6,
        zorder=2,
        label=c.split("_")[-1],
    )

ax_gonio.plot(t_sec, gonio[t[0] : t[-1] + 1], color="0.25", linewidth=1.0, zorder=3)

imu_y0, imu_y1 = ax_imu.get_ylim()
gonio_y0, gonio_y1 = ax_gonio.get_ylim()

# Reserve a band below the IMU signal and above the gonio signal for the
# window brackets, stacked one row per window.
gyro_y0, gyro_y1 = ax_gyro.get_ylim()
imu_band_h = (imu_y1 - imu_y0) * 0.45
gonio_band_h = (gonio_y1 - gonio_y0) * 0.45
gyro_band_h = (gyro_y1 - gyro_y0) * 0.45
ax_imu.set_ylim(imu_y0 - imu_band_h, imu_y1)
ax_gyro.set_ylim(gyro_y0 - gyro_band_h, gyro_y1)
ax_gonio.set_ylim(gonio_y0, gonio_y1 + gonio_band_h)

row_h_imu = imu_band_h / (N_WINDOWS + 0.5)
row_h_gonio = gonio_band_h / (N_WINDOWS + 0.5)

for i in range(N_WINDOWS):
    w_start = start0 + i * DISPLAY_STRIDE
    ctx_start, ctx_end = w_start, w_start + CONTEXT_LENGTH
    fc_start, fc_end = ctx_end, ctx_end + FORECAST_HORIZON

    ctx_x0, ctx_x1 = ctx_start / TARGET_FREQ, ctx_end / TARGET_FREQ
    fc_x0, fc_x1 = fc_start / TARGET_FREQ, fc_end / TARGET_FREQ
    color = colors[i]

    y_imu = imu_y0 - imu_band_h + (i + 0.75) * row_h_imu
    ax_imu.plot([ctx_x0, ctx_x1], [y_imu, y_imu], color=color, lw=3.2, solid_capstyle="butt", zorder=2)
    ax_imu.plot([ctx_x0, ctx_x0], [y_imu - row_h_imu * 0.18, y_imu + row_h_imu * 0.18], color=color, lw=1.6)
    ax_imu.plot([ctx_x1, ctx_x1], [y_imu - row_h_imu * 0.18, y_imu + row_h_imu * 0.18], color=color, lw=1.6)
    ax_imu.annotate(
        f"W{i+1}",
        xy=(ctx_x1 + 0.05, y_imu),
        ha="left",
        va="center",
        fontsize=8.5,
        color=color,
        fontweight="bold",
    )

    y_gonio = gonio_y1 + gonio_band_h - (i + 0.75) * row_h_gonio
    ax_gonio.plot([fc_x0, fc_x1], [y_gonio, y_gonio], color=color, lw=3.2, solid_capstyle="butt", zorder=2)
    ax_gonio.plot([fc_x0, fc_x0], [y_gonio - row_h_gonio * 0.18, y_gonio + row_h_gonio * 0.18], color=color, lw=1.6)
    ax_gonio.plot([fc_x1, fc_x1], [y_gonio - row_h_gonio * 0.18, y_gonio + row_h_gonio * 0.18], color=color, lw=1.6)
    ax_gonio.annotate(
        f"W{i+1}",
        xy=(fc_x0 - 0.05, y_gonio),
        ha="right",
        va="center",
        fontsize=8.5,
        color=color,
        fontweight="bold",
    )

    # Dashed connector linking the end of the context window to the start
    # of its matching forecast window (same colour = same sample boundary).
    fig.add_artist(
        plt.matplotlib.patches.ConnectionPatch(
            xyA=(ctx_x1, y_imu),
            coordsA=ax_imu.transData,
            xyB=(fc_x0, y_gonio),
            coordsB=ax_gonio.transData,
            color=color,
            lw=1.3,
            linestyle=(0, (3, 2)),
            alpha=0.85,
            zorder=5,
        )
    )

ax_imu.set_ylabel("Right thigh\nIMU (normalized)")
ax_imu.set_title("Context window (input)", fontsize=11, loc="left")
ax_gonio.set_ylabel("Right knee\nangle (°)")
ax_gonio.set_title("Forecast window (target)", fontsize=11, loc="left")
ax_gonio.set_xlabel("Time (s)")
ax_imu.set_xlim(t_sec[0], t_sec[-1])
ax_gonio.set_xlim(t_sec[0], t_sec[-1])

handles_a, labels_a = ax_imu.get_legend_handles_labels()
handles_g, labels_g = ax_gyro.get_legend_handles_labels()
ax_imu.legend(
    handles_a + handles_g,
    labels_a + labels_g,
    loc="upper right",
    ncol=2,
    fontsize=8,
    framealpha=0.9,
)

fig.suptitle(
    f"Sliding context-forecast windows (context = {CONTEXT_LENGTH} samples, "
    f"forecast = {FORECAST_HORIZON} samples @ {TARGET_FREQ} Hz)",
    fontsize=11,
)

plt.tight_layout(rect=(0, 0, 1, 0.94))
plt.savefig("img/windowing_diagram.png", dpi=200)
print("Plot successfully saved to img/windowing_diagram.png")
