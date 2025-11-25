import matplotlib.pyplot as plt 
import numpy as np
import os

plt.rcParams['font.family'] = 'DejaVu Serif'

folders = ['../01-100/00-tools', '../02-110/00-tools', '../03-111/00-tools', '../04-031/00-tools']
notations = ['<100>', '<110>', '<111>', '<031>']
for folder in folders:
    dft_file = os.path.join(folder, 'dist-diffenergy-force-DFT')
    gap_file = os.path.join(folder, 'dist-diffenergy-force-GAP')
    dft_txt = np.loadtxt(dft_file, delimiter=',')
    gap_txt = np.loadtxt(gap_file, delimiter=',')

    fig, ax = plt.subplots(1, 2, figsize=(12, 4)) 
    ax[0].plot(dft_txt[:,0], dft_txt[:,1], '^', color='black', label='DFT')
    ax[0].plot(gap_txt[:,0], gap_txt[:,1], '-o', markersize=3, label='GAP')
    ax[0].set_xlabel('Quasi-statically dragged distance (Å)')
    ax[0].set_ylabel('Energy difference (eV)')
    ax[0].grid(True)
    ax[0].legend()

    colors = ['red', 'blue', 'green']
    ax[1].plot(dft_txt[:,0], dft_txt[:,2], '^', color=colors[0], label=r'DFT $F_x$')
    ax[1].plot(gap_txt[:,0], gap_txt[:,2], '-o', markersize=3, color=colors[0] , label=r'GAP $F_x$')
    ax[1].plot(dft_txt[:,0], dft_txt[:,3], '^', color=colors[1], label=r'DFT $F_y$')
    ax[1].plot(gap_txt[:,0], gap_txt[:,3], '-o', markersize=3, color=colors[1], label=r'GAP $F_y$')
    ax[1].plot(dft_txt[:,0], dft_txt[:,4], '^', color=colors[2], label=r'DFT $F_z$')
    ax[1].plot(gap_txt[:,0], gap_txt[:,4], '-o', markersize=3, color=colors[2], label=r'GAP $F_z$')
    ax[1].set_xlabel('Quasi-statically dragged distance (Å)')
    ax[1].set_ylabel('Force on displaced atom (eV/Å)')
    ax[1].grid(True)
    ax[1].legend()

    plt.tight_layout()
    plt.savefig('100.png', dpi=300)





