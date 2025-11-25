import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['font.family'] = 'DejaVu Serif'

# Data for the first plot
x1 = ['H', 'T', 'X', 'B']
Labels1 = ['LDA 128 Silva et al.', 'LDA 128 Moreira et al.', 
           'LDA 144 Holmström et al.', 'LDA+U 64 \u015Apiewak et al.', 'GGA 64 Sueoka et al.', 'GGA 144 Holmström et al.', 'GGA 216 (This work)', 'GAP 216']
df1 = pd.read_csv('formation.csv').fillna(0)

x_pos1 = np.arange(len(x1))  
width1 = 0.1 

fig, axes = plt.subplots(1, 2, figsize=(8, 6), gridspec_kw={'width_ratios': [2.5, 1]})

# Define colors (first 6 are transparent, last 2 are bold and vibrant)
base_colors = plt.cm.tab10.colors  # Get a standard colormap
light_colors = [(c[0], c[1], c[2], 0.5) for c in base_colors[:6]]  # Light & transparent colors
highlight_colors = ['darkblue', 'red']  # Bold, vibrant colors for last two datasets

for i in range(len(Labels1)):
    y1 = df1.iloc[:, i+1]  # Extract column data
    color = light_colors[i] if i < 6 else highlight_colors[i - 6]  # Apply transparency to first 6
    axes[0].bar(x_pos1 + i * width1, y1, width=width1, label=Labels1[i], color=color)

# Formatting the first plot
axes[0].set_xticks(x_pos1 + width1 * (len(Labels1) / 2))
axes[0].set_xticklabels(x1, fontsize=10)  
axes[0].set_xlabel("SIA configuration", fontsize=12)
axes[0].set_ylabel("Formation Energy (eV)", fontsize=12)
axes[0].legend(fontsize=8, loc='upper left', frameon=True, facecolor='white', framealpha=0.8, ncol=2)
axes[0].set_ylim(0, 5.5)
axes[0].tick_params(axis='y', which='both', labelsize=10, length=4, width=1)  
axes[0].grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
axes[0].text(0.9, 0.9, '(a)', transform=axes[0].transAxes,
        fontsize=15, color='black', va='center')

# Data for the second plot
df2 = pd.read_csv('relative.csv')

for i in range(len(Labels1)):
    y2 = df2.iloc[:, i+1]  # Extract column data
    color = light_colors[i] if i < 6 else highlight_colors[i - 6]  # Use the same color scheme
    linewidth = 2 if i >= 6 else 1  # Make the last two lines bolder
    markersize = 6 if i >= 6 else 5  # Make the last two markers larger
    axes[1].plot(np.array(x1), np.array(y2), 'D-', label=Labels1[i], color=color, linewidth=linewidth, markersize=markersize)

# Formatting the second plot
axes[1].set_xticks(np.arange(len(x1)))
axes[1].set_xticklabels(x1, fontsize=10)  
axes[1].set_xlabel("SIA configuration", fontsize=12)
axes[1].set_ylabel("Relative Formation Energy (eV)", fontsize=12)
axes[1].set_ylim(-0.1, 1.7)
axes[1].tick_params(axis='y', which='both', labelsize=10, length=4, width=1)  
axes[1].grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
axes[1].text(0.1, 0.9, '(b)', transform=axes[1].transAxes,
        fontsize=15, color='black', va='center')
# Combine both plots into a single image
plt.tight_layout()
plt.subplots_adjust(wspace=0.2)  # Adjust space between subplots
plt.savefig('Eng_formation_inter.png', dpi=300)
