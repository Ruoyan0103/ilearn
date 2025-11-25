import numpy as np
from quippy.potential import Potential
from ase.io import read, write
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import KernelPCA
from quippy.descriptors import Descriptor
import seaborn as sns

plt.rcParams['font.family'] = 'DejaVu Serif'

def get_pot():
    pot = Potential('xml_label=GAP_2025_3_8_120_22_43_25_989',
                    param_filename='Ge-v10.xml',
                    calc_args='local_gap_variance')
    return pot 

def calc_single_struct_uncertainty_one_file(database: str, pot: Potential, fig, uncertaintyfile: str):
    struct = read(database)
    struct.calc = pot 
    # PKA_pos = struct.positions[1771] # 100 eV case, lammps id = 1772
    PKA_pos = struct.positions[60513]  # 2 keV case,  lammps id = 60514
    local_energy = struct.get_potential_energies() 
    local_sigma = np.sqrt(pot.extra_results['atoms']['local_gap_variance'])  # eV/atom
    ratio = [abs(sigma / eng) * 100 for eng, sigma in zip(local_energy, local_sigma)]
    print(f'average ratio of local_sigma to local_energy: {np.mean(ratio):.2f} %')
    print("PKA_pos: ", PKA_pos)
    print("maximum local energy (eV/atom): ", np.max(local_energy))
    print("minimum local energy (eV/atom): ", np.min(local_energy))
    with open(uncertaintyfile, 'w') as f:
        for item in local_sigma:
            f.write(f"{item}\n")
            
    # fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(3, 1, 1, projection='3d') 
    cmap = plt.cm.YlOrRd
    colors = cmap(np.linspace(0.2, 1.0, 256))
    dark_ylorrd = LinearSegmentedColormap.from_list('trunc_ylorrd', colors)

    def filter(local_sigma, threshold=40):
        high_sigma_list = []
        low_sigma_list = []
        for idx, v in enumerate(local_sigma):
            if v * 1e3 > threshold: # convert to meV/atom
                high_sigma_list.append(idx)
            else:
                low_sigma_list.append(idx)
        return high_sigma_list, low_sigma_list
    
    high_sigma_list, low_sigma_list = filter(local_sigma, threshold=40)
    sizes = [0.05 if idx in low_sigma_list else 10 for idx in range(len(local_sigma))]
    # sizes[60513] = 30  # Highlight PKA atom size
    sc1 = ax1.scatter(struct.positions[:, 0], struct.positions[:, 1], struct.positions[:, 2],
                      s=sizes, c=local_sigma, vmin=min(local_sigma), vmax=max(local_sigma), cmap=dark_ylorrd)
    ax1.scatter(PKA_pos[0], PKA_pos[1], PKA_pos[2], s=20, facecolor='none', edgecolor='black', label='PKA')
    cbar = fig.colorbar(sc1, ax=ax1, pad=0.05, location='right', shrink=0.8)
    cbar.set_label('Local predicted uncertainty (eV/atom)', fontsize=12)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.set_axis_off()
    
    x_min, x_max = np.min(struct.positions[:, 0]), np.max(struct.positions[:, 0])
    y_min, y_max = np.min(struct.positions[:, 1]), np.max(struct.positions[:, 1])
    z_min, z_max = np.min(struct.positions[:, 2]), np.max(struct.positions[:, 2])
    ax1.set_xlim([x_min, x_max])  
    ax1.set_ylim([y_min, y_max])  
    ax1.set_zlim([z_min, z_max])
    # Use text2D for reliable text positioning in 3D plot
    ax1.text2D(
        0.02, 0.98,
        '(a)',  # (a), (c), (e), etc.
        transform=ax1.transAxes,
        verticalalignment='top',
        horizontalalignment='left',
        fontsize=20
    )
    arrow_dir = np.array([0.48167057, 0.30553306, 0.82136655])
    arrow_length = 15  # adjust as needed for visibility
    arrow_vec = arrow_dir / np.linalg.norm(arrow_dir) * arrow_length
    arrow_start = PKA_pos
    ax1.quiver(
        arrow_start[0], arrow_start[1], arrow_start[2],
        arrow_vec[0], arrow_vec[1], arrow_vec[2],
        color='black', linewidth=1, arrow_length_ratio=0.5
    )

    # Second subplot - scatter plot
    ax2 = fig.add_subplot(313)
    sc2 = ax2.scatter(local_energy, local_sigma, s=5, marker='o', c=ratio, vmin=0, vmax=1, cmap='coolwarm')
    # sc2 = ax2.scatter(local_energy, ratio, s=15, marker='o', c=ratio, vmin=min(ratio), vmax=max(ratio), cmap='coolwarm')
    cbar = fig.colorbar(sc2, ax=ax2, pad=0.05, location='right', shrink=0.8)
    cbar.set_label('Relative uncertainty (%)', fontsize=12)
    ax2.set_xlabel('Local energy (eV/atom)', fontsize=12)
    ax2.set_ylabel('Local predicted uncertainty (eV/atom)', fontsize=12)
    ax2.text(
        0.08, 0.95,
        '(c)',  # (b), (d), (f), etc.
        transform=ax2.transAxes,
        verticalalignment='top',
        horizontalalignment='left',
        fontsize=18,
    )
    ax2.axvline(x=-4.52, color='grey', linestyle='--', linewidth=1)
    ax2.axvline(x=-3.64, color='grey', linestyle='--', linewidth=1)

def read_uncertainty(filename: str) -> list:
    with open(filename, 'r') as f:
        loaded_list = [float(line.strip()) for line in f]
    return loaded_list

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

def plot_train_testpoints(
        embedding_train: np.ndarray,
        embedding_test: np.ndarray,
        uncertainty: list,
        fig,  
        method: str='kPCA'
    ):
    # fig, ax = plt.subplots(figsize=(8, 6))  # Single subplot
    ax = fig.add_subplot(3, 1, 2)
    cmap = plt.cm.YlOrRd
    colors = cmap(np.linspace(0.2, 1.0, 256))
    dark_ylorrd = LinearSegmentedColormap.from_list('trunc_ylorrd', colors)
    
    # Plot both on the same axes
    sc1 = ax.scatter(
        embedding_train[0],
        embedding_train[1],
        label='Training',
        s=10,
        alpha=0.5,
        color='lightgray'
    )
    sc2 = ax.scatter(
        embedding_test[0],
        embedding_test[1],
        label='Test',
        c=uncertainty,
        cmap=dark_ylorrd,
        s=10,
    )
    ax.legend(loc='upper right', fontsize=10)
    cbar = fig.colorbar(sc2, ax=ax, pad=0.05, location='right', shrink=0.8)
    cbar.set_label('Local predicted uncertainty (eV/atom)', fontsize=12)
    ax.set_xlabel(f'{method} component 1', fontsize=12)
    ax.set_ylabel(f'{method} component 2', fontsize=12)
    ax.text(
        0.05, 0.95,
        '(b)',  # (b), (d), (f), etc.
        transform=ax.transAxes,
        verticalalignment='top',
        horizontalalignment='left',
        fontsize=18,
    )
    # Set limits based on combined data
    all_x = np.concatenate([embedding_train[0], embedding_test[0]])
    all_y = np.concatenate([embedding_train[1], embedding_test[1]])
    ax.set_xlim(np.min(all_x) * 1.1, np.max(all_x) * 1.1)
    ax.set_ylim(np.min(all_y) * 1.1, np.max(all_y) * 1.1)

####################################################################################
cascadefile = '2keV/berendsen/trajectory-130.xyz'
uncertaintyfile = '2keV/berendsen/trajectory-130-uncertainty.txt'

# from ase.io.lammpsrun import read_lammps_dump_text
# with open('2keV/berendsen/dump.PKA', 'r') as f:
#     db_PKA = read_lammps_dump_text(f, index=130, specorder=['Ge'])
# write(cascadefile, db_PKA, format='extxyz', append=False)

fig = plt.figure(figsize=(6, 15))
pot = get_pot()
calc_single_struct_uncertainty_one_file(cascadefile, pot, fig, uncertaintyfile)

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

data = cascadefile
config_types, descriptors, labels, labels_names = generate_descriptor(data, desc_str)
test_desp = np.vstack(descriptors)
descriptor_transformed = kpca.transform(test_desp)
explained_variance = kpca.eigenvalues_
total_variance = np.sum(explained_variance)
explained_variance_ratio = explained_variance / total_variance
uncertainty = read_uncertainty(uncertaintyfile)
plot_train_testpoints(
    all_desp_transformed.T,
    descriptor_transformed.T,
    uncertainty,
    fig,
    method='kPCA'
)
plt.tight_layout()
plt.savefig('cascade-uncertainty-v2.png', dpi=300)

