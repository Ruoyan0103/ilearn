import numpy as np
import matplotlib.pyplot as plt

# Set font
plt.rcParams['font.family'] = 'DejaVu Serif'
#plt.rcParams['font.size'] = 14  # Increase font size

# File paths and labels
thermal_labels = ['DFT', 'GAP', 'Exp. Reeber et al.']
thermal_colors = ['black', 'red', 'blue']
thermal_files = [
    'data/thermal_expansion_dft.dat',
    'data/thermal_expansion_GAP.dat',
    'exp/ge_thermal_exp_v4'
]

cp_labels = ['DFT', 'GAP', 'Exp. Estermann et al.', 'Exp. Flubacher et al.', 'Exp. Leadbetter et al.']
cp_colors = ['black', 'red', 'blue', 'green', 'orange']
cp_files = [
    'data/cp_polyfit_dft.dat',
    'data/cp_polyfit_GAP.dat',
    'exp/ge_cp_exp_v2',
    'exp/ge_cp_exp_v3',
    'exp/ge_cp_exp_v4'
]

bulk_labels = ['DFT', 'GAP', 'Exp. Yin et al.']
bulk_colors = ['black', 'red', 'blue']
bulk_files = [
    'data/bulk_dft.dat',
    'data/bulk_GAP.dat',
    'exp/ge_bulk_exp_v2'
]

# Create figure with 3 subplots
fig, axes = plt.subplots(3, 1, figsize=(6, 7), sharex=True)

# Plot function
def plot(ax, files, labels, colors, ylabel, markers=None):
    for index, file in enumerate(files):
        if index in [0, 1]:  # DFT and GAP
            txt = np.loadtxt(file)
            temp = txt[:, 0]
            val = txt[:, 1]
            if ylabel == r'$\alpha_L$' + ' (K$^{-1}$)':
                val /= 3  # Scaling for thermal expansion
            elif ylabel == r'$C_{p}$' + ' (J/mol/K)':
                val /= 2  # Scaling for heat capacity
            ax.plot(temp, val, label=labels[index], color=colors[index], linewidth=1)
        else:  # Experimental data
            txt = np.loadtxt(file, skiprows=1)
            if len(txt.shape) == 1:  # If the file has only one value
                temp = 77  
                val = 77
            else:
                temp = txt[:, 0]
                val = txt[:, 1]
            if ylabel == r'$\alpha_L$' + ' (K$^{-1}$)':
                val /= 1e6  # Scaling for thermal expansion
            ax.scatter(temp, val, marker=markers[index-2], facecolors='none', edgecolors=colors[index], 
                       label=labels[index], s=50, linewidth=1)

    ax.legend(loc='best', framealpha=1, fontsize=8)
    ax.minorticks_on()
    ax.tick_params(axis='both', which='major', length=5)
    ax.tick_params(axis='both', which='minor', length=3)
    ax.grid(linestyle='--', alpha=0.7)
    ax.set_ylabel(ylabel, fontsize=10)
    subplot_labels = ['(a)', '(b)', '(c)']
    ax.text(0.5, 0.7, subplot_labels[ax.get_subplotspec().rowspan.start], transform=ax.transAxes,
        fontsize=14 , va='top', ha='left')

# Plot each dataset

plot(axes[0], thermal_files, thermal_labels, thermal_colors, r'$\alpha_L$' + ' (K$^{-1}$)', markers=['*'])
plot(axes[1], cp_files, cp_labels, cp_colors, r'$C_{p}$' + ' (J/mol/K)', markers=['^', 's', 'o'])
plot(axes[2], bulk_files, bulk_labels, bulk_colors, r"$B$ (GPa)", markers=['*'])

# X-axis label only on the last plot
axes[-1].set_xlabel('Temperature (K)', fontsize=12)

plt.tight_layout()
plt.savefig('combined_thermal.png', dpi=300)
