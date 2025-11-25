import numpy as np
import matplotlib.pyplot as plt
from quippy.descriptors import Descriptor
from ase.io import read
import seaborn as sns 
from sklearn.decomposition import KernelPCA, PCA
from sklearn.manifold import TSNE
from matplotlib.colors import LinearSegmentedColormap
from monty.serialization import loadfn
import seaborn as sns
import pandas as pd

plt.rcParams['font.family'] = 'DejaVu Serif'  

def generate_descriptor(
        database: str,
        desc_str: str,
        atoms_or_structures: str='atoms'
    ) -> tuple[dict, list, np.ndarray, list]:
    structures = read(database, index=':')
    desc = Descriptor(desc_str)
    config_types = {}
    descriptors = []
    labels = []
    labels_names = []

    for s in structures:
        d = desc.calc_descriptor(s)
        descriptors.append(d)
        if 'config_type' not in s.info:
            s.info['config_type'] = 'Test'
        ctype = s.info['config_type']
        if ctype not in config_types:
            config_types[ctype] = []
        config_types[ctype].append(s)

    for idx, (key, struct_list) in enumerate(config_types.items()):
        num_structs = len(struct_list)
        num_atoms = sum(len(s) for s in struct_list)
        if atoms_or_structures == 'atoms':
            labels.extend([idx] * num_atoms)
        elif atoms_or_structures == 'structures':
            labels.extend([idx] * num_structs)
        if key == 'betaTin':
            key = r"$\beta$-Sn"
        elif key == 'split':
            key = 'interstitial-X'
        elif key == 'bond':
            key = 'interstitial-B'
        elif key == 'tet':
            key = 'interstitial-T'
        elif key == 'hex':
            key = 'interstitial-H'
        if key not in labels_names:
            labels_names.append(key)
    return config_types, descriptors, np.array(labels), labels_names

def plot_points(
        embedding: np.ndarray, 
        labels: np.ndarray, 
        label_names: list, 
        markers: list,
        ax,
        method: str='kPCA'):
    
    def tab20_spread(n=16):
        cmap = plt.cm.get_cmap('tab20')
        all_colors = cmap(np.arange(20) / 20)
        order = [1, 3, 5, 6, 14, 4, 2, 8, 10, 12, 16, 18, 0, 7, 
                 13, 15, 17, 19]
        spread_colors = all_colors[order][:n]
        return spread_colors

    colors = tab20_spread()
    for idx, (label_name, marker) in enumerate(zip(label_names, markers)):
        ax.scatter(
                embedding[0, labels == idx],
                embedding[1, labels == idx],
                label=label_name,
                marker=marker,
                s=30,
                facecolors='none',
                edgecolors=colors[idx % len(colors)]
            )
    ax.text(
        0.85, 0.95,
        '(a)',
        transform=ax.transAxes,
        verticalalignment='top',
        horizontalalignment='left',
        fontsize=15,
        fontname='DejaVu Serif'
    )
    ax.legend(
        fontsize=8,
        loc='lower right',
        markerscale=1.2,
        ncol=2,
        borderpad=0.1,      
        labelspacing=0.1,   
        handlelength=1,     
        handletextpad=0.4  
    )
    ax.set_xlabel(f'kPCA component 1', fontsize=11)
    ax.set_ylabel(f'kPCA component 2', fontsize=11)
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)

def get_energies_and_plot(database: str, database2: str, ax):
    energies = []
    structures = read(database, index=':') 

    for struct in structures:
        energy = struct.get_potential_energy(force_consistent=True) / len(struct)
        energies.append(energy)

    sns.set_theme(style="whitegrid")
    sns.histplot(energies, color="red", binwidth=0.05, ax=ax)

    ax.set_xlabel('Energy per atom (eV)', fontdict={'family': 'DejaVu Serif', 'size': 11})
    ax.set_ylabel('Number of structures', fontdict={'family': 'DejaVu Serif', 'size': 11})
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.text(
        0.05, 0.95,
        '(b)',
        transform=ax.transAxes,
        verticalalignment='top',
        horizontalalignment='left',
        fontsize=15,
        fontname='DejaVu Serif'
    )
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)

    ax_inset = ax.inset_axes([0.55, 0.55, 0.42, 0.42])
    energies = []
    structures = read(database2, index=':') 

    for struct in structures:
        energy = struct.get_potential_energy(force_consistent=True) / len(struct)
        energies.append(energy)
    sns.set_theme(style="whitegrid")
    sns.histplot(energies, color="red", binwidth=0.05, ax=ax_inset)
    # Remove x and y labels
    ax_inset.set_xlabel('')
    ax_inset.set_ylabel('')

    # Set tick parameters with font
    for tick in ax_inset.get_xticklabels():
        tick.set_fontsize(8)
        tick.set_fontname('DejaVu Serif')
    for tick in ax_inset.get_yticklabels():
        tick.set_fontsize(8)
        tick.set_fontname('DejaVu Serif')

    # Optional: keep grid style if needed
    ax_inset.grid(True, linestyle='--', alpha=0.7)

#########################################################################
desc_str = "soap_turbo l_max=8 alpha_max={8} atom_sigma_r={0.5} atom_sigma_t={0.5} \
            atom_sigma_r_scaling={0.} atom_sigma_t_scaling={0.} \
            zeta=4, rcut_hard=5.0 rcut_soft=4.5 basis=poly3gauss scaling_mode=polynomial add_species=F \
            amplitude_scaling={1.0} n_species=1 species_Z={32} central_index=1 \
            radial_enhancement=1 compress_mode=trivial central_weight={1.0} \
            config_type_n_sparse={dimer:10:diamond:20:fcc:10:hcp:20:bcc:10:st12:10:betaTin:10:bc8:20:hd:10:phonon:100:vacancy:100:tet:100:split:100:bond:100}"

data = 'train_liquid_mod-interstitial.xyz'
config_types, descriptors, labels, labels_names = generate_descriptor(data, desc_str)
markers = ['o', 's', '^', 'v', '<', '>', 'p', 'h', '*', 'P', 'H', 'D', 'd', '.']
kpca = KernelPCA(n_components=2, kernel='poly', degree=2, coef0=1, random_state=42)
all_desp = np.vstack(descriptors)
kpca.fit(all_desp)
all_desp_transformed = kpca.transform(all_desp)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
plot_points(all_desp_transformed.T, labels, labels_names, markers, ax[0], method='kPCA')
get_energies_and_plot('../../train_liquid_mod-interstitial-nodimer.xyz', '../../train_liquid_mod-interstitial-v2.xyz', ax[1])
plt.tight_layout()
plt.savefig('distribution.png', dpi=300)






 





























