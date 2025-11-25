import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams['font.family'] = 'DejaVu Serif'
plt.rcParams['font.size'] = 8  # Increase font size for readability

folders = ['../01-100/00-tools', '../02-110/00-tools', '../03-111/00-tools', '../04-031/00-tools']
notations = [r'$\langle 100 \rangle$', r'$\langle 110 \rangle$', r'$\langle 111 \rangle$', r'$\langle 031 \rangle$']

fig, axes = plt.subplots(4, 2, figsize=(6, 7.5))  # 4 rows (notations) × 2 columns (energy, force)

colors = ['darkgreen', 'darkorange', 'darkblue']
markersize = 1
linewidth = 1

for idx, (folder, notation) in enumerate(zip(folders, notations)):
    dft_file = os.path.join(folder, 'dist-diffenergy-force-DFT')
    gap_file = os.path.join(folder, 'dist-diffenergy-force-GAP')
    
    dft_txt = np.loadtxt(dft_file, delimiter=',')
    gap_txt = np.loadtxt(gap_file, delimiter=',')

    # Energy subplot
    ax_energy = axes[idx, 0]
    ax_energy.plot(dft_txt[:, 0], dft_txt[:, 1], '^', color='black', label='DFT', markersize=3)
    ax_energy.plot(gap_txt[:, 0], gap_txt[:, 1], '-o', color='red', label='GAP', markersize=2.5, linewidth=linewidth)
    
    ax_energy.set_ylabel('Energy difference (eV)', fontsize=7)
    ax_energy.grid(True, linestyle='--', alpha=0.7)
    ax_energy.minorticks_on()
    ax_energy.text(0.1, 0.85, notation, transform=ax_energy.transAxes, fontsize=7.5, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))

    # Force subplot
    ax_force = axes[idx, 1]
    if idx == 0:
        ax_force.plot(dft_txt[:, 0], dft_txt[:, 2], '^', color=colors[0], label=r'DFT $F_x$', markersize=markersize+2)
        ax_force.plot(gap_txt[:, 0], gap_txt[:, 2], 'o', color=colors[0], label=r'GAP $F_x$', linestyle='-', markersize=markersize+2, linewidth=linewidth, fillstyle='none')
        ax_force.plot(dft_txt[:, 0], dft_txt[:, 3], 'D', color=colors[1], label=r'DFT $F_y$ = $F_z$', markersize=markersize+2)
        ax_force.plot(gap_txt[:, 0], gap_txt[:, 3], 's', color=colors[1], label=r'GAP $F_y$ = $F_z$', linestyle='-', markersize=markersize+2, linewidth=linewidth, fillstyle='none')
        # ax_force.plot(dft_txt[:, 0], dft_txt[:, 4], '*', color=colors[2], label=r'DFT $F_z$', markersize=markersize+4)
        # ax_force.plot(gap_txt[:, 0], gap_txt[:, 4], 'v', color=colors[2], label=r'GAP $F_z$', linestyle='-', markersize=markersize+1, linewidth=linewidth, fillstyle='none')
        ax_force.minorticks_on()
        ax_force.set_ylabel('Force on displaced atom (eV/Å)', fontsize=7)
        ax_force.grid(True, linestyle='--', alpha=0.7)
    elif idx == 1:
        ax_force.plot(dft_txt[:, 0], dft_txt[:, 2], '^', color=colors[0], label=r'DFT $F_x$ = $F_y$', markersize=markersize+2)
        ax_force.plot(gap_txt[:, 0], gap_txt[:, 2], 'o', color=colors[0], label=r'GAP $F_x$ = $F_y$', linestyle='-', markersize=markersize+2, linewidth=linewidth, fillstyle='none')
        # ax_force.plot(dft_txt[:, 0], dft_txt[:, 3], 'D', color=colors[1], label=r'DFT $F_y$', markersize=markersize+2)
        # ax_force.plot(gap_txt[:, 0], gap_txt[:, 3], 's', color=colors[1], label=r'GAP $F_y$', linestyle='-', markersize=markersize+2, linewidth=linewidth, fillstyle='none')
        ax_force.plot(dft_txt[:, 0], dft_txt[:, 4], '*', color=colors[2], label=r'DFT $F_z$', markersize=markersize+4)
        ax_force.plot(gap_txt[:, 0], gap_txt[:, 4], 'v', color=colors[2], label=r'GAP $F_z$', linestyle='-', markersize=markersize+1, linewidth=linewidth, fillstyle='none')
        ax_force.minorticks_on()
        ax_force.set_ylabel('Force on displaced atom (eV/Å)', fontsize=7)
        ax_force.grid(True, linestyle='--', alpha=0.7)
    elif idx == 2:
        ax_force.plot(dft_txt[:, 0], dft_txt[:, 2], '^', color=colors[0], label=r'DFT $F_x$ = $F_y$ = $F_z$', markersize=markersize+2)
        ax_force.plot(gap_txt[:, 0], gap_txt[:, 2], 'o', color=colors[0], label=r'GAP $F_x$ = $F_y$ = $F_z$', linestyle='-', markersize=markersize+2, linewidth=linewidth, fillstyle='none')
        # ax_force.plot(dft_txt[:, 0], dft_txt[:, 3], 'D', color=colors[1], label=r'DFT $F_y$', markersize=markersize+2)
        # ax_force.plot(gap_txt[:, 0], gap_txt[:, 3], 's', color=colors[1], label=r'GAP $F_y$', linestyle='-', markersize=markersize+2, linewidth=linewidth, fillstyle='none')
        # ax_force.plot(dft_txt[:, 0], dft_txt[:, 4], '*', color=colors[2], label=r'DFT $F_z$', markersize=markersize+4)
        # ax_force.plot(gap_txt[:, 0], gap_txt[:, 4], 'v', color=colors[2], label=r'GAP $F_z$', linestyle='-', markersize=markersize+1, linewidth=linewidth, fillstyle='none')
        ax_force.minorticks_on()
        ax_force.set_ylabel('Force on displaced atom (eV/Å)', fontsize=7)
        ax_force.grid(True, linestyle='--', alpha=0.7)
    else:
        ax_force.plot(dft_txt[:, 0], dft_txt[:, 2], '^', color=colors[0], label=r'DFT $F_x$', markersize=markersize+2)
        ax_force.plot(gap_txt[:, 0], gap_txt[:, 2], 'o', color=colors[0], label=r'GAP $F_x$', linestyle='-', markersize=markersize+2, linewidth=linewidth, fillstyle='none')
        ax_force.plot(dft_txt[:, 0], dft_txt[:, 3], 'D', color=colors[1], label=r'DFT $F_y$', markersize=markersize+2)
        ax_force.plot(gap_txt[:, 0], gap_txt[:, 3], 's', color=colors[1], label=r'GAP $F_y$', linestyle='-', markersize=markersize+2, linewidth=linewidth, fillstyle='none')
        ax_force.plot(dft_txt[:, 0], dft_txt[:, 4], '*', color=colors[2], label=r'DFT $F_z$', markersize=markersize+4)
        ax_force.plot(gap_txt[:, 0], gap_txt[:, 4], 'v', color=colors[2], label=r'GAP $F_z$', linestyle='-', markersize=markersize+1, linewidth=linewidth, fillstyle='none')
        ax_force.minorticks_on()
        ax_force.set_ylabel('Force on displaced atom (eV/Å)', fontsize=7)
        ax_force.grid(True, linestyle='--', alpha=0.7)

    
    #ax_force.text(0.05, 0.90, notation, transform=ax_force.transAxes, fontsize=16, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))
    
    if idx == 3:
        ax_force.legend(fontsize=6, loc='lower left', ncol=3)
    else:
        ax_force.legend(fontsize=6, loc='lower left')
    ax_energy.legend(fontsize=6, loc='lower right')
    ax_force.tick_params(axis='both', labelsize=6.5)
    ax_energy.tick_params(axis='both', labelsize=6.5)
    if idx == 3:
        ax_force.set_xlabel('Quasi-statically dragged distance (Å)', fontsize=7)
        ax_energy.set_xlabel('Quasi-statically dragged distance (Å)', fontsize=7)


plt.tight_layout()
plt.savefig('combined_path.png', dpi=600)
