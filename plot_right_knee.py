import pandas as pd
import matplotlib.pyplot as plt

file_path = "data/raw/ENABL3S/AB185/Processed/AB185_Circuit_003_post.csv"
df = pd.read_csv(file_path)

plt.figure(figsize=(10, 6))
plt.plot(df['Right_Knee'], label='Right Knee Angle', color='blue')
plt.title('Right Knee Angle Over Time - AB185 Circuit 003', fontsize=14)
plt.xlabel('Time (samples)', fontsize=12)
plt.ylabel('Angle (degrees)', fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('right_knee_plot.png')
print("Plot successfully saved to right_knee_plot.png")
