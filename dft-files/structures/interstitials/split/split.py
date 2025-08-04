from ase.build import bulk, make_supercell
from ase.io import read, write
import numpy as np
from ase import Atoms

a0 = 5.762
Ge_cubic = bulk('Ge', 'diamond', a=a0, cubic=True)
size = 3 
matrix = [[size, 0, 0], [0, size, 0], [0, 0, size]]
Ge_sup = make_supercell(Ge_cubic, matrix)
del Ge_sup[1] # atom id 2
#del Ge_sup[118]

alpha = (2.495/2*np.sqrt(3)/2/a0)
interstitial_position1 = np.array([(0.25-alpha)*a0, (0.25-alpha)*a0, 0.25*a0])
#interstitial_position1 = np.array([(0.5-alpha)*a0*3, (0.5+alpha)*a0*3, (2/3)*a0*3]) 
interstitial_atom1 = Atoms('Ge', positions=[interstitial_position1])
interstitial_position2 = np.array([(0.25+alpha)*a0, (0.25+alpha)*a0, 0.25*a0])
#interstitial_position2 = np.array([(0.5+alpha)*a0*3, (0.5-alpha)*a0*3, (2/3)*a0*3]) 
interstitial_atom2 = Atoms('Ge', positions=[interstitial_position2])
combined_structure = Ge_sup + interstitial_atom1 + interstitial_atom2

#write('Ge.split', combined_structure, format='lammps-data')
write('POSCARtest', combined_structure, format='vasp')

