import numpy as np
import matplotlib.pyplot as plt
import glob, os

plt.rcParams['font.family'] = 'DejaVu Serif'

def plot_rdf(ax, files, Labels, Colors, ylow, yupp, num):
    for index, file in enumerate(files):
        data = np.loadtxt(file, skiprows=2)
        r_values = data[:, 0]
        rdf_data = data[:, 1]
        if 'exp' in file:
            ax.plot(r_values, rdf_data, '--', label=Labels[index], color=Colors[index], linewidth=1.5)
        elif 'MD' in file:
            ax.plot(r_values, rdf_data, label=Labels[index], color=Colors[index], linewidth=2)
        elif 'DFT' in file:
            ax.plot(r_values, rdf_data, label=Labels[index], color=Colors[index], linewidth=1.5)

    ax.legend(fontsize=12, loc='upper right', frameon=True, framealpha=0.8)
    ax.set_xlabel(r"$r$ (Å)", fontsize=14)
    ax.set_ylabel(r"g($r$)", fontsize=14)
    ax.set_xlim(0, 10)
    ax.set_ylim(ylow, yupp)
    ax.tick_params(axis='both', labelsize=12)
    if num == 1:
        ax.text(1.5, 1.8, '(a)', fontsize=15, ha='center')
    else:
        ax.text(1.6, 5.2, '(b)', fontsize=15, ha='center')
    #ax.set_yticks(np.arange(ylow, yupp + interval, interval))

# Create subplots (1 row, 2 columns)
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# First plot (Amorphous)
Labels1 = ['GAP', r'$upper$ Exp.', r'$lower$ Exp.']
Colors1 = ['red', '#348ABD', '#800080']
files1 = ['../01-amorphous/01-MD/liquid.dat', '../01-amorphous/02-exp/liquid-highbound.dat', 
          '../01-amorphous/02-exp/liquid-lowbound.dat']
plot_rdf(axes[1], files1, Labels1, Colors1, 0, 6, 2)

# Second plot (Liquid)
Labels2 = ['DFT', 'GAP', 'Exp.']
Colors2 = ['black', 'red', '#2CA02C']
files2 = ['../02-liquid/01-DFT/NVT-222/liquid.dat', '../02-liquid/02-MD/liquid.dat', '../02-liquid/03-exp/liquid.dat']
plot_rdf(axes[0], files2, Labels2, Colors2, 0, 2.3, 1)

plt.tight_layout()
plt.savefig("rdf_comparison.png", dpi=300)
