import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'DejaVu Serif'

# File paths
exp_file = '04-Literature/exp'  
phonon_band_files = ['01-dft/555-80K/raw-data.txt', '02-lammps/666/raw-data.txt']
dos_files = ['01-dft/555-80K/total_dos.dat', '02-lammps/666/total_dos.dat']

# Settings
phonon_colors = ['black', 'red']
dos_colors = ['black', 'red']
phonon_labels = ['DFT', 'GAP']
dos_labels = ['DFT', 'GAP']
xaxis_labels = ['$\\Gamma$', 'X', 'K', '$\\Gamma$', 'L']

def split_phonon_band_data(raw_data_file, tolerance):
    with open(raw_data_file, 'r') as f:
        f.readline()
        segs = np.array(f.readline().strip().strip('#').strip().split(), dtype=float)
    
    
    ch, freq = np.loadtxt(raw_data_file, unpack=True)
    lbs = np.where(abs(ch - segs[0]) < tolerance)[0]
    hbs = np.where(abs(ch - segs[-1]) < tolerance)[0]
    ch_split, freq_split = [], []
    
    for l, h in zip(lbs, hbs):
        ch_split.append(ch[l:h+1])
        freq_split.append(freq[l:h+1])
    
    return ch_split, freq_split, segs

# Initialize figure
fig, axs = plt.subplots(1, 2, figsize=(8, 6), gridspec_kw={'width_ratios': [4, 1], 'wspace': 0})

# Plot experimental data as scatter plot (Left Panel)
x_values, y_values = [], []
with open(exp_file, 'r') as file:
    for line in file:
        if line.strip():
            x, y = map(float, line.strip().split(','))
            x_values.append(x)
            y_values.append(y)

# Plot phonon band structure (Left Panel)
for ci, raw_data in enumerate(phonon_band_files):
    chs, freqs, segs = split_phonon_band_data(raw_data, 1.0e-6)
    index = 0
    for ch, freq in zip(chs, freqs):
        index += 1
        if index == 6:
            axs[0].plot(ch, freq, phonon_colors[ci], label=phonon_labels[ci])
        else:
            axs[0].plot(ch, freq, phonon_colors[ci])
axs[0].scatter(x_values, y_values, facecolors='none', edgecolors='blue', marker='o', s=20, label='Exp.')
axs[0].set_xlim(segs[0], segs[-1])
axs[0].set_ylim(-0.5, 9.5)
axs[0].set_xticks(segs)
axs[0].tick_params(axis='both', which='major', labelsize=12)
axs[0].set_xticklabels(xaxis_labels)
axs[0].set_ylabel('Frequency (THz)', fontsize=14)
axs[0].grid(True, linestyle='--', alpha=0.6)
axs[0].legend(fontsize=14, bbox_to_anchor=(0.4, 0.2))

# Plot phonon DOS (Right Panel)
for index, file in enumerate(dos_files):
    data = np.loadtxt(file)
    frequencies, dos = data[:, 0], data[:, 1]
    axs[1].plot(dos, frequencies, color=dos_colors[index], lw=1.5, label=dos_labels[index])
axs[1].set_ylim(-0.5, 9.5)
axs[1].grid(True, linestyle='--', alpha=0.6)
axs[1].tick_params(axis='x', which='both', bottom=False, labelbottom=False, labelsize=14)
axs[1].tick_params(axis='y', which='both', left=True, labelleft=False, labelsize=14)
axs[1].set_xlabel('DOS', fontsize=14)
axs[1].tick_params(axis='x', which='both', bottom=True, labelsize=12)

plt.tight_layout()
plt.savefig('combined_phonon_plot.png', dpi=300)

