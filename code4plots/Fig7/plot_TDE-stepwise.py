import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
plt.rcParams['font.family'] = 'DejaVu Serif'

def plot_TDE(ax, TDE_file):
    txt = np.loadtxt(TDE_file, skiprows=2, usecols=(0, 1, 2, 3, 4, 5, 6, 7))
    azimuthal = txt[:, 4]   # Azimuthal angles in radians
    polar = txt[:, 5]       # Polar angles in radians
    energy = txt[:, 7]      # Energy values in eV

    print(f'minimum energy: {np.min(energy)} eV')
    print(f'maximum energy: {np.max(energy)} eV')
     
    #fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    phi_grid = np.linspace(0, np.pi/4, 1000)
    theta_grid = np.linspace(0, np.deg2rad(54.7), 1000)
    phi_grid2, theta_grid2 = np.meshgrid(phi_grid, theta_grid)
    
    # Mask invalid points in the interpolation grid
    # rcos(theta) = l         (1)
    # rsin(theta)cos(phi) = h (2)
    # rsin(theta)sin(phi) = k (3)
    # The maximum polar angle is when h = l, so we can find the maximum polar angle
    # using (1) and (2), max_polar = np.arctan((h/l) / cos(phi)) = np.arctan(1 / cos(phi))
    max_polar_grid = np.arctan(1 / np.cos(phi_grid2))
    valid_mask = theta_grid2 <= max_polar_grid    # boolean

    # Interpolation with masking
    energy_grid = griddata(
        (azimuthal, polar), energy, (phi_grid2, theta_grid2), method='linear'
    )
    energy_grid[~valid_mask] = np.nan  # Set invalid regions to NaN
 
    ax.set_thetalim(0, np.pi/4)
    ax.set_rlim(0, np.deg2rad(54.7))
    ax.set_yticks(np.radians([0, 15, 30, 45, 54.7]))            # Set ticks in polar
    ax.tick_params(axis='y', labelsize=8)
    ax.set_yticklabels(['0°', '15°', '30°', '45°', '54.7°'])
    ax.grid(True, linewidth=0.3, color='#D3D3D3')

    ax.set_xticks(np.radians([0, 5, 10, 15, 20, 25, 30, 35, 40, 45]))  # Set ticks in radians
    ax.set_xticklabels(['0°', '5°', '10°', '15°', '20°', '25°', '30°', '35°', '40°', '45°'], fontsize=8)
    contourf = ax.contourf(phi_grid2, theta_grid2, energy_grid, levels=10, cmap='viridis', alpha=0.89)
    contour = ax.contour(phi_grid2, theta_grid2, energy_grid, levels=10, colors='black', linewidths=0.18)
    scatter = ax.scatter(azimuthal, polar, c='#D3D3D3', s=0.8, edgecolors='#D3D3D3')
    
    # Text 1: Along the arc ("Azimuthal")
    angle = np.pi / 8                                # Position along the arc (midway between 0 and pi/4)
    radius = np.deg2rad(54.7)                        # Maximum polar angle
    text_radius = radius * 1.2                     # Slightly farther from the arc
    ax.text(
        angle, text_radius, 'Azimuthal angle, φ', 
        rotation=np.degrees(angle - np.pi / 2),      # Align with the arc's tangent
        ha='center', va='center', fontsize=9, color='black',
    )

    # Text 2: Below the center
    ax.text(
        0.2, 0, '[001]',  # Below the origin
        ha='right', va='bottom', fontsize=9, color='black'
    )

    # Text 3: Below the 54.7° polar angle
    ax.text(
        -0.16, np.deg2rad(45), '[101]',  # Below the arc
        ha='center', va='center', fontsize=9, color='black'
    )

    # Text 4: Above the 45° azimuthal angle
    ax.text(
        0.87, np.deg2rad(53.5), '[111]',  # Slightly above the 45° azimuthal angle
        ha='center', va='center', fontsize=9, color='black'
    )

    # Text 5: xlable
    ax.text(
        -0.3, np.deg2rad(27), 'Polar angle, θ',  # Adjust x and y to position the text
        ha='center', va='center', fontsize=9, color='black'
    )

    if 'MEAM' in TDE_file:
        ax.text(
            0.1, 0.8,
            '(b)',
            ha='center', va='center',
            fontsize=11, color='black',
            transform=ax.transAxes
        )
    else:
        ax.text(
            0.1, 0.8,
            '(a)',
            ha='center', va='center',
            fontsize=11, color='black',
            transform=ax.transAxes
        )

    vmin, vmax = np.nanmin(energy_grid), np.nanmax(energy_grid)
    tick_step = 3.5
    ticks = np.arange(np.floor(vmin), np.ceil(vmax) + tick_step, tick_step)
    char = fig.colorbar(contourf, ax=ax, pad=0.15, shrink=0.85, ticks=ticks)
    char.set_label('Threshold displacement energy (eV)', fontsize=9)        
    char.ax.tick_params(
        labelsize=8,          # Smaller font size for numbers
        length=2,            # Shorter tick marks
        width=0.5            # Thinner tick marks
    )
    plt.savefig('polar_combine-step.png', dpi=300)


def plot_threeDir(ax):
    # Data
    dft = [19.5, 28.5, 11.5]   # 111 open direction
    dft_std = [1.5, 0, 1.5]
    gap = [11.3, 26.8, 6.3]
    meam = [18.8, 22.1, 18.3]
    x = [r'$\langle 100 \rangle$', r'$\langle 110 \rangle$', r'$\langle 111 \rangle$']

    # Create figure with custom size (width, height in inches)
    fig, ax = plt.subplots(figsize=(6, 6))  # Adjust these dimensions as needed

    # Plot data
    ax.errorbar(x, dft, yerr=dft_std, label='DFT E Holmström', fmt='o-', capsize=5, color='black')
    ax.plot(x, gap, label='GAP', marker='o', color='red')
    ax.plot(x, meam, label='MEAM', marker='^', color='green')

    # Customize axes
    # ax.set_xticks(x)
    # ax.set_xlabel('Crystal direction', fontsize=14)
    # ax.set_ylabel('TDE (eV)', fontsize=14)
    # ax.tick_params(axis='both', which='major', labelsize=12)

    # Add legend and grid
    ax.legend(fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.7)

    # Adjust layout and save
    # plt.tight_layout()
    # plt.savefig('threeDir.png', dpi=300, bbox_inches='tight')


fig, ax = plt.subplots(2, 1, subplot_kw={'projection': 'polar'}, figsize=(5, 6))
GAP_TDE_file = 'GAP_TDE_v3.txt'
MEAM_TDE_file = 'MEAM_TDE_v2.txt'
plt.tight_layout()
plot_TDE(ax[0], GAP_TDE_file)
plot_TDE(ax[1], MEAM_TDE_file)









