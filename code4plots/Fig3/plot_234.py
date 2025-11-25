import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

################################### Figure arrangement ###################################
# fig = plt.figure(figsize=(10, 8))
# gs = gridspec.GridSpec(3, 2, width_ratios=[1, 1])
# ax_left_top = fig.add_subplot(gs[0, 0])   

# inner = gridspec.GridSpecFromSubplotSpec(
#     1, 2, subplot_spec=gs[1:3, 0], width_ratios=[4, 1], wspace=0
# )
# ax_band = fig.add_subplot(inner[0])
# ax_dos = fig.add_subplot(inner[1])
# ax_r1 = fig.add_subplot(gs[0, 1])
# ax_r2 = fig.add_subplot(gs[1, 1])
# ax_r3 = fig.add_subplot(gs[2, 1])


fig = plt.figure(figsize=(10, 8))

gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])

ax_left_top = fig.add_subplot(gs[0, 0])
inner = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=gs[1, 0], width_ratios=[4, 1], wspace=0
)
ax_band = fig.add_subplot(inner[0])
ax_dos = fig.add_subplot(inner[1])

right_inner = gridspec.GridSpecFromSubplotSpec(
    3, 1, subplot_spec=gs[:, 1], hspace=0.05
)
ax_r1 = fig.add_subplot(right_inner[0])
ax_r2 = fig.add_subplot(right_inner[1])
ax_r3 = fig.add_subplot(right_inner[2])

plt.rcParams['font.family'] = 'DejaVu Serif'
labelsize = 12
legendsize = 8
textsize = 14
######################################### Figure1 E-V #####################################
def plot_EV(files, ax):
    color_map = plt.get_cmap('plasma', len(files))
    labels = ['dia','dia', 'hd','hd', 'bc8','bc8', 'st12','st12',
              r'$\beta$-Sn', r'$\beta$-Sn', 'hcp', 'hcp', 'fcc', 'fcc', 'bcc', 'bcc' ]
    x = []
    y = []
    for i, f in enumerate(files):
        c = color_map(i / len(files))
        with open (f, 'r') as file:
            for line in file:
                w = line.split(',')
                x.append(float(w[1]))
                y.append(float(w[2]))

        if 'DFT' in f:
            ax.plot(x, y, 'D', markersize=4, color=c)
        if 'GAP' in f:
            ax.plot(x, y, markersize=0.8, color=c, label=labels[i])
        x = []
        y = []
    ax.legend(fontsize=legendsize, loc='lower left', frameon=True, facecolor='white', framealpha=0.8)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax.set_xlabel('Volume (Å³/atom)', fontsize=labelsize, labelpad=10, fontfamily='DejaVu Serif')
    ax.set_ylabel('Energy (eV/atom)', fontsize=labelsize, labelpad=10, fontfamily='DejaVu Serif')
    ax.set_xlim(15, 32)
    ax.set_ylim(-4.55, -3.95)
    ax.text(0.8, 0.9, '(a)', transform=ax.transAxes,
        fontsize=textsize, va='top', ha='left')
    for label in ax.get_xticklabels():
        label.set_fontfamily('DejaVu Serif')
    for label in ax.get_yticklabels():
        label.set_fontfamily('DejaVu Serif')

######################################### Figure2 phonon #####################################
def plot_phonon(exp_file, phonon_band_files, dos_files, ax_band, ax_dos):
    phonon_colors = ['black', 'red']
    dos_colors = ['black', 'red']
    phonon_labels = ['DFT', 'GAP']
    dos_labels = ['DFT', 'GAP']
    xaxis_labels = ['$\\Gamma$', 'X', 'K', '$\\Gamma$', 'L']

    # experimental data
    x_values, y_values = [], []
    with open(exp_file, 'r') as file:
        for line in file:
            if line.strip():
                x, y = map(float, line.split(','))
                x_values.append(x)
                y_values.append(y)
    # helper
    def split_phonon_band_data(raw_data_file, tolerance):
        with open(raw_data_file, 'r') as f:
            f.readline()
            segs = np.array(f.readline().strip().strip('#').split(), float)
        ch, freq = np.loadtxt(raw_data_file, unpack=True)
        lbs = np.where(abs(ch - segs[0]) < tolerance)[0]
        hbs = np.where(abs(ch - segs[-1]) < tolerance)[0]
        ch_split, freq_split = [], []
        for l, h in zip(lbs, hbs):
            ch_split.append(ch[l:h+1])
            freq_split.append(freq[l:h+1])
        return ch_split, freq_split, segs

    # ---- BAND STRUCTURE (LEFT) ----
    for ci, raw_data in enumerate(phonon_band_files):
        chs, freqs, segs = split_phonon_band_data(raw_data, 1e-6)
        branch_index = 0
        for ch, freq in zip(chs, freqs):
            branch_index += 1
            if branch_index == 6:
                ax_band.plot(ch, freq, phonon_colors[ci], label=phonon_labels[ci])
            else:
                ax_band.plot(ch, freq, phonon_colors[ci])

    ax_band.scatter(x_values, y_values, facecolors='none', edgecolors='blue',
                    marker='o', s=20, label='Exp.')
    ax_band.set_xlim(segs[0], segs[-1])
    ax_band.set_ylim(-0.5, 9.5)
    ax_band.set_xticks(segs)
    ax_band.set_xticklabels(xaxis_labels, fontfamily='DejaVu Serif')
    ax_band.set_ylabel('Frequency (THz)', fontsize=labelsize, fontfamily='DejaVu Serif')
    ax_band.grid(True, linestyle='--', alpha=0.6)
    ax_band.legend(fontsize=legendsize, loc='lower left')
    ax_band.text(0.7, 0.7, '(b)', transform=ax_band.transAxes,
        fontsize=textsize, va='top', ha='left')

    # ---- DOS (RIGHT) ----
    for ci, file in enumerate(dos_files):
        data = np.loadtxt(file)
        freq, dos = data[:, 0], data[:, 1]
        ax_dos.plot(dos, freq, color=dos_colors[ci], lw=1.5,
                    label=dos_labels[ci])
    ax_dos.set_ylim(-0.5, 9.5)
    ax_dos.grid(True, linestyle='--', alpha=0.6)
    ax_dos.set_xlabel('DOS', fontsize=labelsize, fontfamily='DejaVu Serif')
    ax_dos.tick_params(axis='y', which='both', labelleft=False)
    ax_dos.tick_params(axis='x', which='both', labelbottom=False)


######################################### Figure3 thermal #####################################
def plot_thermal(ax, files, labels, colors, ylabel, markers=None):
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
    # the third ax
    if ylabel == r'$B$' + ' (GPa)':
        ax.legend(loc='lower left', framealpha=1, fontsize=legendsize+1)
    else:
        ax.legend(loc='best', framealpha=1, fontsize=legendsize+1)
    ax.minorticks_on()
    ax.tick_params(axis='both', which='major', length=5)
    ax.tick_params(axis='both', which='minor', length=3)
    ax.grid(linestyle='--', alpha=0.7)
    ax.set_ylabel(ylabel, fontsize=10, fontfamily='DejaVu Serif')
    for label in ax.get_yticklabels():
        label.set_fontfamily('DejaVu Serif')
    for label in ax.get_xticklabels():
        label.set_fontfamily('DejaVu Serif')
    subplot_labels = ['(c)', '(d)', '(e)']
    ax.text(0.5, 0.65, subplot_labels[ax.get_subplotspec().rowspan.start], transform=ax.transAxes,
        fontsize=textsize, va='top', ha='left')


######################################### For Plotting #####################################
files = ['01-dia/DFT', '01-dia/GAP', '06-hd/DFT', '06-hd/GAP', 
        '04-bc8/DFT', '04-bc8/GAP','07-st12/DFT', '07-st12/GAP', 
        '08-beta/DFT', '08-beta/GAP', '02-fcc/DFT', '02-fcc/GAP', 
        '05-hcp/DFT', '05-hcp/GAP', '03-bcc/DFT', '03-bcc/GAP']
org_folder = '../Fig2'
new_files = []
for file in files:
    new_file = os.path.join(org_folder, file)
    new_files.append(new_file)
plot_EV(new_files, ax_left_top)

exp_file = '../Fig3/04-Literature/exp'
phonon_band_files = ['../Fig3/01-dft/555-80K/raw-data.txt', '../Fig3/02-lammps/666/raw-data.txt']
dos_files = ['../Fig3/01-dft/555-80K/total_dos.dat', '../Fig3/02-lammps/666/total_dos.dat']
plot_phonon(exp_file, phonon_band_files, dos_files, ax_band, ax_dos)

thermal_labels = ['DFT', 'GAP', 'Exp. Reeber et al.']
thermal_colors = ['black', 'red', 'blue']
thermal_files = [
    '../Fig4/data/thermal_expansion_dft.dat',
    '../Fig4/data/thermal_expansion_GAP.dat',
    '../Fig4/exp/ge_thermal_exp_v4'
]
cp_labels = ['DFT', 'GAP', 'Exp. Estermann et al.', 'Exp. Flubacher et al.', 'Exp. Leadbetter et al.']
cp_colors = ['black', 'red', 'blue', 'green', 'orange']
cp_files = [
    '../Fig4/data/cp_polyfit_dft.dat',
    '../Fig4/data/cp_polyfit_GAP.dat',
    '../Fig4/exp/ge_cp_exp_v2',
    '../Fig4/exp/ge_cp_exp_v3',
    '../Fig4/exp/ge_cp_exp_v4'
]
bulk_labels = ['DFT', 'GAP', 'Exp. Yin et al.']
bulk_colors = ['black', 'red', 'blue']
bulk_files = [
    '../Fig4/data/bulk_dft.dat',
    '../Fig4/data/bulk_GAP.dat',
]
plot_thermal(ax_r1, thermal_files, thermal_labels, thermal_colors, r'$\alpha_L$' + ' (K$^{-1}$)', markers=['*'])
plot_thermal(ax_r2, cp_files, cp_labels, cp_colors, r'$C_{p}$' + ' (J/mol/K)', markers=['^', 's', 'o'])
plot_thermal(ax_r3, bulk_files, bulk_labels, bulk_colors, r'$B$' + ' (GPa)', markers=['*'])

plt.tight_layout()
plt.savefig('combined_bulk_prop.png', dpi=300)
