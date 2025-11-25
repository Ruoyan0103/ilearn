import matplotlib.pyplot as plt 
import numpy as np
import os
from ase.build import bulk

def cal_dist(atom1_pos, atom2_pos, interval, number_struct_from, number_struct):
    '''
    iinterval: denominator
    number_struct_from: some of the distances are invalid, remove those part (default=0)
    number_struct: numerator
    '''
    dist = np.linalg.norm(atom1_pos - atom2_pos)
    dist_list = []
    for i in range(number_struct_from, number_struct):
        inter_dist = i*(dist)/interval
        dist_list.append(inter_dist)
    return dist_list

a0 = 5.762
size = 2
Ge_cubic = bulk('Ge', 'diamond', a=a0, cubic=True)
Ge_sup = Ge_cubic * size
atom1_pos = Ge_sup[0].position
atom2_pos = Ge_sup[14].position

dist_list = cal_dist(atom1_pos, atom2_pos, 26, 0, 5)
directory = '../02-cal'
energies = np.loadtxt(os.path.join(directory, 'energies'))
energies_i =  np.atleast_1d(energies)
plt.plot(dist_list, [e-energies_i[0] for e in energies_i], '^', color='black', label='DFT')


dist_list = cal_dist(atom1_pos, atom2_pos, 26, 14, 21)
directory = '../02-cal'
energies_2 = np.loadtxt(os.path.join(directory, 'energies_2'))
energies_2_i = np.atleast_1d(energies_2)
plt.plot(dist_list, [e-energies_i[0] for e in energies_2_i], '^', color='black')


dist_list = cal_dist(atom1_pos, atom2_pos, 20, 0, 17)
energies_gap = np.loadtxt(os.path.join(directory, 'energies-GAP'))
energies_gap_i =  np.atleast_1d(energies_gap)
plt.plot(dist_list, [e-energies_gap_i[0] for e in energies_gap_i],'-o', markersize=3, label='GAP')

plt.xlabel('Quasi-statically dragged distance (Å)')
plt.ylabel('Energy difference (eV)')
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig(os.path.join(directory, 'energies.png'), dpi=300)

