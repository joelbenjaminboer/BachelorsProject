import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_path = "data/raw/ENABL3S/AB185/Processed/AB185_Circuit_003_post.csv"
df = pd.read_csv(file_path)

# Right_Heel_Contact holds the row index of each right heel-strike event;
# one full gait cycle runs from a heel-strike to the next one on the same side.
heel_strikes = df["Right_Heel_Contact"].dropna().astype(int).tolist()
start, end = heel_strikes[0], heel_strikes[1]

cycle = df.iloc[start:end]
pct_gait_cycle = np.linspace(0, 100, len(cycle))

plt.figure(figsize=(10, 6))
plt.plot(pct_gait_cycle, cycle["Right_Thigh_Ax"], label="Ax", color="green")
plt.plot(pct_gait_cycle, cycle["Right_Thigh_Ay"], label="Ay", color="red")
plt.plot(pct_gait_cycle, cycle["Right_Thigh_Az"], label="Az", color="blue")
plt.title("Right Thigh Accelerometer Over One Gait Cycle - AB185 Circuit 003")
plt.xlabel("gait (%)")
plt.ylabel("Accelerometer (m/s²)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("accel_gait_cycle_plot.png")
print("Plot successfully saved to accel_gait_cycle_plot.png")
