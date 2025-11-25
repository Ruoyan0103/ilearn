import matplotlib.pyplot as plt
import re
import numpy as np
from scipy.interpolate import make_interp_spline
from crt_zbl import ZBL

# Set font and initialize ZBL
plt.rcParams['font.family'] = 'DejaVu Serif'
zbl = ZBL(32, 32)

def plotAll(files1, files2, outfile):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # Create 1x2 subplot layout
    axes = axes.ravel()  # Flatten the array of axes for easier access
    def plot_files(ax, files, use_log=False):
        location = 'upper right'
        for f in files:
            x, y = [], []
            with open(f, 'r') as file:
                for line in file:
                    words = re.split(r'[,\s]+', line.strip())
                    if len(words) >= 2:  # Ensure line has at least two values
                        x.append(float(words[0]))
                        y.append(float(words[1]))

            if 'dmol' in f:
                ax.plot(x, y, ':o', color='orange', label='DMol', markersize=3, linewidth=2)
            if 'dft1' in f:
                ax.plot(x, y, 'D', color='black', label='DFT', markersize=4, linewidth=2)
            if 'dft2' in f:
                ax.plot(x, y, 'D', color='black', label='DFT', markersize=4, linewidth=2)
            if 'zbl' in f:
                ax.plot(x, y, '-.', color='blue', label=r'$V_{pair}$', markersize=3, linewidth=2)
            if 'GAP' in f:
                ax.plot(x, y, '-', color='red', label='GAP', markersize=3, linewidth=2)

        if use_log:
            ax.set_yscale('log')  # Apply log scale if needed
            location='lower left'
        ax.set_xlabel(r'Distance between atoms $r_{ij}$ (Å)', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(fontsize=14, loc=location)
        

    # Plot the first set of files on the first subplot
    plot_files(axes[0], files1, use_log=True)
    axes[0].set_ylabel('Total Energy (eV)', fontsize=14)
    axes[0].text(0.1, 0.5, '(a)', transform=axes[0].transAxes, fontsize=20, va='top')

    # Create inset axes for the second plot inside ax[0]
    ax_inset = axes[0].inset_axes([0.5, 0.5, 0.48, 0.48])  # [x, y, width, height] in fraction of ax[0]

    # Function to load and filter data
    def load_and_filter(file, max_distance):
        data = np.loadtxt(file)
        mask = data[:, 0] <= max_distance
        return data[mask, 0], data[mask, 1]

    # File paths and labels for the inset plot
    dmol1 = '01-test/reference/dmol1'
    GAP1 = '01-test/00-energy/GAP1'
    GAP2 = '01-test/00-energy2/GAP'
    labels = [r'$V_{pair}$', 'zbl_dmol', 'Dmol']
    colors = ['blue', 'grey', 'orange']

    # Load and process data for the inset plot
    rs_gap1, efile_gap1 = load_and_filter(GAP1, 1.5)
    rs_gap1, efile_gap1 = rs_gap1[::15], efile_gap1[::15]
    ezbl = np.array([zbl.e_zbl(r) for r in rs_gap1])
    gap_zbl = np.abs(efile_gap1 - ezbl) / ezbl * 100
    ax_inset.scatter(rs_gap1, gap_zbl, label=labels[0], color=colors[0], alpha=0.4, s=30)
    x_smooth = np.linspace(min(rs_gap1), max(rs_gap1), 300)
    spl = make_interp_spline(rs_gap1, gap_zbl, k=3)
    ax_inset.plot(x_smooth, spl(x_smooth), '--', color=colors[0], linewidth=1.5)

    rs_dmol, efile_dmol = load_and_filter(dmol1, 1.5)
    rs_gap2, efile_gap2 = load_and_filter(GAP2, 1.5)
    ezbl = np.array([zbl.e_zbl(r) for r in rs_dmol])
    dmol_zbl = np.abs(efile_dmol - ezbl) / efile_dmol * 100
    gap_dmol = np.abs(efile_gap2 - efile_dmol) / efile_dmol * 100
    ax_inset.scatter(rs_dmol, gap_dmol, color=colors[2], alpha=0.4, s=30, label=labels[2])
    x_smooth = np.linspace(min(rs_dmol), max(rs_dmol), 300)
    spl = make_interp_spline(rs_dmol, gap_dmol, k=3)
    ax_inset.plot(x_smooth, spl(x_smooth), '--', color=colors[2], linewidth=1.5)

    # Customize the inset plot
    ax_inset.tick_params(axis='both', which='major', labelsize=10)
    ax_inset.set_ylabel('Relative Error (%)', fontsize=10)
    ax_inset.set_xlabel(r'$r_{ij}$ (Å)', fontsize=10)
    ax_inset.grid(True, linestyle='--', alpha=0.6)
    ax_inset.legend(fontsize=10)

    # Plot the second set of files on the second subplot
    plot_files(axes[1], files2, use_log=False)
    axes[1].set_ylabel('Total Energy (eV)', fontsize=14)
    axes[1].text(0.2, 0.5, '(b)', transform=axes[1].transAxes, fontsize=20, va='top')

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)

# Example usage
files1 = ['01-test/00-energy/GAP1', '01-test/reference/dmol1', '01-test/reference/dft1', '01-test/reference/zbl1']
files2 = ['01-test/00-energy/GAP2', '01-test/reference/dft2', '01-test/reference/zbl2']
plotAll(files1, files2, 'dimer-eng.png')
