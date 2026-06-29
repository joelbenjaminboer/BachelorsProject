import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 1. Set presentation-friendly visual style
sns.set_theme(style="whitegrid")
sns.set_context("talk") 

# 2. Input your final RMSE results
# Baseline LSTM = Earlier calculated averages
# Combined Split = Your provided metrics
# Custom Transformer = The newly averaged subject metrics
data = {
    'Locomotor State': [
        'Walk', 'Walk',
        'Ramp Up', 'Ramp Up',
        'Ramp Down', 'Ramp Down',
        'Stair Up', 'Stair Up',
        'Stair Down', 'Stair Down'
    ],
    'Architecture': [
        'Encoder', 'TCN baseline', 
        'Encoder', 'TCN baseline', 
        'Encoder', 'TCN baseline', 
        'Encoder', 'TCN baseline',
        'Encoder', 'TCN baseline',
    ],
    'RMSE': [
        # Walk
        15.04, 25.09, 
        # Ramp Up
        11.97, 22.69,  
        # Ramp Down
        11.91, 23.47,  
        # Stair Up
        17.49, 25.03,
        # Stair Down
        15.18, 25.71
    ]
}

df = pd.DataFrame(data)

# 3. Create the figure 
plt.figure(figsize=(16, 8))

# 4. Generate the grouped bar chart
ax = sns.barplot(
    data=df, 
    x='Locomotor State', 
    y='RMSE', 
    hue='Architecture',
    palette=["#d72222", "#378de3"] # Blue for Transformer, Green for Combined, Grey for Baseline
)

# 5. Customize titles and labels
plt.title('Kinematic Prediction Error Across Gait Conditions', pad=20, fontweight='bold')
plt.xlabel('', labelpad=15) 
plt.ylabel('RMSE', labelpad=15)

# Set the y-limit to fit the highest baseline value with some headroom for annotations
plt.ylim(0, 32.0) 

# 6. Annotate the bars with exact RMSE values
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.2f'), 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha = 'center', va = 'center', 
                xytext = (0, 10), 
                textcoords = 'offset points',
                fontsize=11)

# 7. Clean up the legend
plt.legend(title='', loc='upper left', framealpha=0.9)
sns.despine(left=True) 

# 8. Save the high-resolution plot
plt.tight_layout()
plt.savefig('symposium_final_rmse_plot.png', dpi=300)
plt.show()