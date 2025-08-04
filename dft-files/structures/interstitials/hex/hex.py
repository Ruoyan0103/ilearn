from ase.build import bulk, make_supercell
from ase.io import read, write
import numpy as np
from ase import Atoms

a0 = 5.7620
Ge_cubic = bulk('Ge', 'diamond', a=a0, cubic=True)
matrix = [[3, 0, 0], [0, 3, 0], [0, 0, 3]]
Ge_sup = make_supercell(Ge_cubic, matrix)
interstitial_position = np.array([0.625*a0, 0.625*a0, 0.625*a0])
interstitial_position = np.array([(0.625/3+0.5-1/6)*a0*3, (0.625/3+0.5-1/6)*a0*3, (0.625/3+0.5-1/6)*a0*3])
interstitial_atom = Atoms('Ge', positions=[interstitial_position])
combined_structure = Ge_sup + interstitial_atom

write('Ge.hex', combined_structure, format='lammps-data')
write('POSCAR', combined_structure, format='vasp')




